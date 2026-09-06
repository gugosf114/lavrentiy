"""
What I Meant — Cloud Function Backend Proxy
Deploy to GCP Cloud Functions (Python 3.11+).

Authenticates users via Firebase Auth, checks subscription tier,
rate-limits, then calls OpenAI with the server-side API key.

Endpoint: POST /
Headers: Authorization: Bearer <firebase_id_token>
Body: {
    "raw": "so um like the thing is we need to uh get the report",
    "tone": "professional",
    "layer": 2,
    "situation": "default",
    "mode": "FAST",
    "profile": {...}
}
"""

import json
import logging
import os
import time

import functions_framework
from google.cloud import firestore
from firebase_admin import auth, initialize_app

from reconstruct import reconstruct_intent
from audio_backend import AudioRequestError, prepare_audio_request
from learning_backend import LearningResponseError, learn_from_pairs
from billing_backend import (
    BillingVerificationError,
    PACKAGE_NAME as BILLING_PACKAGE_NAME,
    PRODUCT_ID as BILLING_PRODUCT_ID,
    account_hash,
    acknowledge_with_google,
    fetch_subscription_with_google,
    subscription_is_entitled,
    token_hash,
    verify_with_google,
)
from billing_events import BillingEventError, decode_pubsub_cloud_event
from quota_backend import plan_audio_usage, plan_cloud_usage
from reviewer_access import reviewer_email_is_allowed
from user_data_backend import delete_cloud_account, make_json_safe

# Structured logging — Cloud Run / Functions parses JSON lines on stdout into
# Cloud Logging fields. INFO level for normal flow, ERROR for failures.
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _emit(level, **fields):
    """Emit a single JSON-shaped log line for Cloud Logging structured search."""
    try:
        logging.log(level, json.dumps(fields, default=str))
    except Exception:
        # Never let logging itself break a request.
        pass

# Initialize Firebase Admin (uses default GCP credentials)
try:
    initialize_app()
except ValueError:
    pass  # Already initialized

db = firestore.Client()

# Tier definitions. One cloud dictation consumes one audio slot and one general
# slot; keeping those counters parallel prevents double-charging while bounding
# both provider paths. L4 also consumes its smaller high-cost monthly allowance.
TIERS = {
    "invite": {"max_layer": 1, "monthly_limit": 0, "l4_monthly_limit": 0,
               "name": "Local / Free"},
    # The Layer 4 sub-cap is gone. It was set when L4 meant Sonnet extended
    # thinking; the Cloud Function serves L4 on gpt-4o, where a take costs
    # about two cents against one for L2/L3. Three hundred takes all at L4
    # costs roughly $6 against a $9.99 subscription, so the cap was rationing
    # the clinical disfluency layer — the reason the product exists — to
    # protect a margin that was never at risk.
    "basic": {"max_layer": 4, "monthly_limit": 300, "l4_monthly_limit": 300,
              "name": "WiM Cloud"},
    "pro": {"max_layer": 4, "monthly_limit": 999999,
            "l4_monthly_limit": 999999, "name": "Pro"},
}

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "3600",
}


def verify_token(request):
    """Extract and verify Firebase ID token from Authorization header.
    Returns (decoded_claims, error_response)."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, (json.dumps({"error": "Missing Authorization header"}), 401, CORS_HEADERS)

    token = auth_header.split("Bearer ")[1].strip()
    try:
        decoded = auth.verify_id_token(token)
        return decoded, None
    except Exception as e:
        return None, (json.dumps({"error": f"Invalid token: {str(e)[:100]}"}), 401, CORS_HEADERS)


def get_user_tier(uid, reviewer=False):
    """Get user's subscription tier from Firestore. Default to 'invite'."""
    ref = db.collection("wim_users").document(uid)
    doc = ref.get()
    data = doc.to_dict() if doc.exists else {}
    if reviewer:
        # Play reviewers cannot create a purchase or trial during app review.
        # A verified, explicitly allowlisted Google account receives durable
        # full access on its first authenticated request instead.
        if data.get("tier") != "pro" or not data.get("reviewer_access"):
            grant = {
                "tier": "pro",
                "reviewer_access": True,
                "reviewer_granted_at": firestore.SERVER_TIMESTAMP,
            }
            if not doc.exists:
                grant.update({
                    "created": firestore.SERVER_TIMESTAMP,
                    "usage_period": time.strftime("%Y-%m", time.gmtime()),
                })
            ref.set(grant, merge=True)
        return "pro"
    if doc.exists:
        tier = data.get("tier", "invite")
        # New subscription entitlements expire server-side even if an old app
        # keeps a stale local boolean. Preserve any pre-launch legacy/manual
        # tier that has a different product ID so existing test access is not
        # accidentally destroyed during migration.
        if (tier == "basic" and data.get("billing_product_id") == BILLING_PRODUCT_ID
                and float(data.get("billing_expiry_ts", 0) or 0) <= time.time()):
            return "invite"
        return tier
    # First-time user: create doc with invite tier
    ref.set({
        "tier": "invite",
        "created": firestore.SERVER_TIMESTAMP,
        "usage_period": time.strftime("%Y-%m", time.gmtime()),
    })
    return "invite"


def check_rate_limit(uid, tier_config, layer=None):
    """Atomically consume one general cloud slot and, for L4, one L4 slot."""
    ref = db.collection("wim_users").document(uid)
    monthly_limit = tier_config["monthly_limit"]
    l4_limit = tier_config["l4_monthly_limit"]
    @firestore.transactional
    def _txn(transaction):
        snapshot = ref.get(transaction=transaction)
        data = snapshot.to_dict() if snapshot.exists else {}
        ok, remaining, error_code, update_data = plan_cloud_usage(
            data, monthly_limit, l4_limit, layer=layer)
        if not ok:
            return False, 0, error_code
        transaction.set(ref, update_data, merge=True)
        return True, remaining, None

    ok, remaining, error_code = _txn(db.transaction())
    if not ok:
        is_l4_error = error_code == "l4_monthly_quota_reached"
        return False, 0, (json.dumps({
            "error": "Monthly L4 limit reached" if is_l4_error else "Monthly cloud limit reached",
            "error_code": error_code,
            "limit": l4_limit if is_l4_error else monthly_limit,
            "tier": tier_config["name"],
        }), 429, CORS_HEADERS)
    return True, remaining, None


def check_audio_rate_limit(uid, tier_config):
    """Separate audio quota so one dictation does not consume two user credits.

    Transcription and reconstruction are two HTTP calls for one user action.
    Charging both against the general counter would silently cut every advertised
    allowance in half; this parallel counter bounds audio cost without doing so.
    """
    ref = db.collection("wim_users").document(uid)
    monthly_limit = tier_config["monthly_limit"]
    @firestore.transactional
    def _txn(transaction):
        snapshot = ref.get(transaction=transaction)
        data = snapshot.to_dict() if snapshot.exists else {}
        ok, remaining, update_data = plan_audio_usage(
            data, monthly_limit)
        if not ok:
            return False, 0
        transaction.set(ref, update_data, merge=True)
        return True, remaining

    ok, remaining = _txn(db.transaction())
    if not ok:
        return False, 0, (json.dumps({
            "error": "Monthly cloud-audio limit reached",
            "error_code": "monthly_audio_quota_reached",
            "limit": monthly_limit,
            "tier": tier_config["name"],
        }), 429, CORS_HEADERS)
    return True, remaining, None


_JSON_CORS = {**CORS_HEADERS, "Content-Type": "application/json"}
_ALLOWED_PROFILE_KEYS = {
    "trigger_words", "onset_weights", "covert_profile",
    "filler_words", "vocabulary", "corrections",
}
_EXPORT_VISIBLE_KEYS = {
    "trigger_words", "onset_weights", "covert_profile",
    "filler_words", "vocabulary", "corrections", "sync_ts", "created",
    "tier", "billing_product_id", "billing_order_id", "billing_verified_at",
    "billing_expiry_ts", "billing_state",
}
_RETRIABLE_EXC_NAMES = {"APITimeoutError", "APIConnectionError", "RateLimitError"}


def _action_sync_profile(uid, tier_config, body):
    """Lavrentiy desktop pushes learned profile to Firestore."""
    raw = body.get("profile", {})
    profile_data = {k: v for k, v in raw.items() if k in _ALLOWED_PROFILE_KEYS}
    profile_data["sync_ts"] = time.time()
    db.collection("wim_users").document(uid).set(profile_data, merge=True)
    return (json.dumps({"ok": True}), 200, _JSON_CORS)


def _action_export_data(uid, tier_config, body):
    """GDPR: export user's stored data, stripping internal billing/quota state."""
    doc = db.collection("wim_users").document(uid).get()
    full = doc.to_dict() if doc.exists else {}
    user_visible = make_json_safe(
        {k: v for k, v in full.items() if k in _EXPORT_VISIBLE_KEYS}
    )
    return (json.dumps({"ok": True, "data": user_visible}), 200, _JSON_CORS)


def _action_delete_data(uid, tier_config, body):
    """GDPR/Play policy: delete all cloud data and the Firebase account."""
    deleted = delete_cloud_account(db, auth, uid)
    return (json.dumps({"ok": True, "deleted": True, **deleted}),
            200, _JSON_CORS)


def _action_billing_status(uid, tier_config, body):
    doc = db.collection("wim_users").document(uid).get()
    data = doc.to_dict() if doc.exists else {}
    tier = get_user_tier(uid)
    return (json.dumps({
        "ok": True,
        "unlocked": tier in ("basic", "pro"),
        "tier": tier,
        "product_id": data.get("billing_product_id"),
        "expires_at": data.get("billing_expiry_ts"),
        "monthly_limit": TIERS.get(tier, TIERS["invite"])["monthly_limit"],
        "l4_monthly_limit": TIERS.get(tier, TIERS["invite"])["l4_monthly_limit"],
    }), 200, _JSON_CORS)


def _action_verify_purchase(uid, tier_config, body):
    """Verify, bind, grant, and acknowledge the current cloud subscription."""
    purchase_token = (body.get("purchase_token") or "").strip()
    product_id = (body.get("product_id") or "").strip()
    try:
        purchase, google_session = verify_with_google(purchase_token, product_id)
        external_id = (purchase.get("externalAccountIdentifiers") or {}).get(
            "obfuscatedExternalAccountId")
        if external_id != account_hash(uid):
            raise BillingVerificationError("Purchase belongs to a different account", 403)

        # A grant must never outlive an unacknowledged purchase: Play refunds
        # unacknowledged purchases after its acknowledgement window. Ack first;
        # a retry can safely finish the idempotent Firestore grant afterward.
        if purchase.get("acknowledgementState") == "ACKNOWLEDGEMENT_STATE_PENDING":
            acknowledge_with_google(google_session, purchase_token, product_id)

        line_item = purchase["_wim_line_item"]
        expiry_ts = purchase["_wim_expiry_ts"]
        order_id = line_item.get("latestSuccessfulOrderId") or purchase.get("latestOrderId")
        purchase_ref = db.collection("wim_subscription_tokens").document(
            token_hash(purchase_token))
        user_ref = db.collection("wim_users").document(uid)

        @firestore.transactional
        def _grant(transaction):
            existing = purchase_ref.get(transaction=transaction)
            if existing.exists and existing.to_dict().get("uid") != uid:
                return False
            transaction.set(purchase_ref, {
                "uid": uid,
                "product_id": BILLING_PRODUCT_ID,
                "order_id": order_id,
                "subscription_state": purchase.get("subscriptionState"),
                "expiry_ts": expiry_ts,
                "verified_at": firestore.SERVER_TIMESTAMP,
            }, merge=True)
            transaction.set(user_ref, {
                "tier": "basic",
                "billing_product_id": BILLING_PRODUCT_ID,
                "billing_order_id": order_id,
                "billing_state": purchase.get("subscriptionState"),
                "billing_expiry_ts": expiry_ts,
                "billing_token_hash": token_hash(purchase_token),
                "billing_verified_at": firestore.SERVER_TIMESTAMP,
            }, merge=True)
            return True

        if not _grant(db.transaction()):
            raise BillingVerificationError("Purchase token was already claimed", 409)

    except BillingVerificationError as e:
        _emit(logging.WARNING, event="billing_verify_rejected", uid=uid,
              status=e.status, error=str(e)[:200])
        return (json.dumps({"error": str(e)}), e.status, CORS_HEADERS)
    except Exception as e:
        _emit(logging.ERROR, event="billing_verify_failed", uid=uid,
              exception=type(e).__name__, error=str(e)[:300])
        return (json.dumps({"error": "Purchase verification temporarily unavailable"}),
                503, CORS_HEADERS)

    _emit(logging.INFO, event="billing_entitlement_granted", uid=uid,
          product_id=product_id, expiry_ts=purchase["_wim_expiry_ts"])
    return (json.dumps({
        "ok": True,
        "unlocked": True,
        "tier": "basic",
        "product_id": BILLING_PRODUCT_ID,
        "expires_at": purchase["_wim_expiry_ts"],
        "monthly_limit": TIERS["basic"]["monthly_limit"],
        "l4_monthly_limit": TIERS["basic"]["l4_monthly_limit"],
    }), 200, _JSON_CORS)


def _action_command(uid, tier_config, body):
    """Command Mode: highlight + voice command → transformed text.
    Re-uses rate-limit + tier infrastructure but skips the heavy reconstruction
    prompt — this is a free-form text transform, not disfluency reconstruction.
    """
    source = (body.get("source") or "").strip()
    command = (body.get("command") or "").strip()
    if not source or not command:
        return (json.dumps({"error": "Missing 'source' or 'command' field"}), 400, CORS_HEADERS)

    ok, remaining, rate_err = check_rate_limit(uid, tier_config)
    if not ok:
        return rate_err

    from reconstruct import client as openai_client
    if openai_client is None:
        return (json.dumps({"error": "Backend OpenAI client not configured"}), 500, CORS_HEADERS)

    try:
        system_prompt = (
            "You are a text transformation assistant. The user highlighted some text "
            "and spoke a command to modify it. Apply the command and return ONLY the "
            "transformed text, nothing else. Preserve the meaning. Do not add "
            "explanations, quotes, or prefixes."
        )
        user_content = f"TEXT:\n{source}\n\nCOMMAND: {command}"
        resp = openai_client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        transformed = (resp.choices[0].message.content or "").strip()
        return (json.dumps({
            "transformed": transformed,
            "tier": tier_config["name"],
            "remaining": remaining,
        }), 200, _JSON_CORS)
    except Exception as e:
        return (json.dumps({"error": f"Transform failed: {str(e)[:200]}"}), 500, CORS_HEADERS)


def _action_complete_partial(uid, tier_config, body):
    """Mid-block bridging: a partial utterance (speaker froze mid-sentence) ->
    up to 3 completion candidates the bubble shows as a tap row. Shared by
    Lavrentiy desktop and WiM Android signed-in users."""
    partial = (body.get("partial_text") or body.get("raw") or "").strip()
    if not partial:
        return (json.dumps({"error": "Missing 'partial_text' field"}), 400, CORS_HEADERS)

    ok, remaining, rate_err = check_rate_limit(uid, tier_config)
    if not ok:
        return rate_err

    from reconstruct import complete_partial
    t_call = time.time()
    try:
        candidates = complete_partial(
            partial,
            tone=body.get("tone", "casual"),
            language_code=body.get("language_code", "en"),
        )
    except Exception as e:
        latency_ms = round((time.time() - t_call) * 1000)
        _emit(logging.ERROR, event="complete_partial_failed", uid=uid,
              latency_ms=latency_ms, exception=type(e).__name__, error=str(e)[:300])
        return (json.dumps({
            "error": f"Completion failed: {type(e).__name__}",
            "tier": tier_config["name"],
        }), 500, CORS_HEADERS)

    _emit(logging.INFO, event="complete_partial_ok", uid=uid,
          latency_ms=round((time.time() - t_call) * 1000), n=len(candidates))
    return (json.dumps({
        "candidates": candidates,
        "tier": tier_config["name"],
        "remaining": remaining,
    }), 200, _JSON_CORS)


def _action_transcribe_audio(uid, tier_config, body):
    """Authenticated mobile audio transcription.

    WiM sends the recording here so release builds never need an OpenAI
    secret on the phone — as base64 JSON (older builds) or as a raw
    multipart `file` part (2026-09-06 builds).  `whisper-1` keeps verbose
    segment confidence for L4; `gpt-4o-transcribe` is the faster normal
    cloud path.
    """
    try:
        kwargs, audio_bytes_len, model = prepare_audio_request(
            body, audio_bytes=body.get("_audio_bytes"))
    except AudioRequestError as e:
        return (json.dumps({"error": str(e)}), e.status, CORS_HEADERS)

    ok, remaining, rate_err = check_audio_rate_limit(uid, tier_config)
    if not ok:
        return rate_err

    from reconstruct import client as openai_client
    if openai_client is None:
        return (json.dumps({"error": "Backend OpenAI client not configured"}), 500, CORS_HEADERS)

    t_call = time.time()
    try:
        response = openai_client.audio.transcriptions.create(**kwargs)
        payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    except Exception as e:
        latency_ms = round((time.time() - t_call) * 1000)
        retriable = _classify_exception(e)
        _emit(logging.ERROR, event="transcribe_audio_failed", uid=uid,
              latency_ms=latency_ms, exception=type(e).__name__,
              error=str(e)[:300], retriable=retriable)
        return (json.dumps({
            "error": f"Transcription failed: {type(e).__name__}",
            "retriable": retriable,
        }), 503 if retriable else 500, CORS_HEADERS)

    text = (payload.get("text") or "").strip()
    segments = payload.get("segments") or []
    words = payload.get("words") or []
    _emit(logging.INFO, event="transcribe_audio_ok", uid=uid, model=model,
          latency_ms=round((time.time() - t_call) * 1000),
          audio_bytes=audio_bytes_len, segments=len(segments), words=len(words),
          envelope="raw" if body.get("_audio_bytes") is not None else "wrapped")
    out = {
        "text": text,
        "segments": segments,
        "model": model,
        "remaining": remaining,
    }
    if words:
        out["words"] = words
    return (json.dumps(out), 200, _JSON_CORS)


def _action_learn_profile(uid, tier_config, body):
    """Run Layer 3 background learning with the server-side OpenAI key."""
    if tier_config["max_layer"] < 3:
        return (json.dumps({
            "error": "Layer 3 profile learning requires WiM Cloud",
            "tier": tier_config["name"],
        }), 403, _JSON_CORS)
    pairs = body.get("pairs", "")
    if not isinstance(pairs, str) or not pairs.strip():
        return (json.dumps({"error": "Missing transcription pairs"}), 400, _JSON_CORS)

    ok, remaining, rate_err = check_rate_limit(uid, tier_config, 3)
    if not ok:
        return rate_err

    from reconstruct import MODEL as reconstruction_model
    from reconstruct import client as openai_client
    t_call = time.time()
    try:
        learnings = learn_from_pairs(
            openai_client,
            reconstruction_model,
            pairs[:12000],
        )
    except LearningResponseError as e:
        _emit(logging.ERROR, event="learn_profile_invalid", uid=uid,
              latency_ms=round((time.time() - t_call) * 1000), error=str(e)[:300])
        return (json.dumps({"error": str(e)}), 502, _JSON_CORS)
    except Exception as e:
        retriable = _classify_exception(e)
        _emit(logging.ERROR, event="learn_profile_failed", uid=uid,
              latency_ms=round((time.time() - t_call) * 1000),
              exception=type(e).__name__, error=str(e)[:300], retriable=retriable)
        return (json.dumps({
            "error": f"Profile learning failed: {type(e).__name__}",
            "retriable": retriable,
        }), 503 if retriable else 500, _JSON_CORS)

    _emit(logging.INFO, event="learn_profile_ok", uid=uid,
          latency_ms=round((time.time() - t_call) * 1000))
    return (json.dumps({
        "learnings": learnings,
        "remaining": remaining,
    }), 200, _JSON_CORS)


def _classify_exception(e):
    """Return True if the exception is retriable per OpenAI SDK conventions.
    Introspects by name + status_code to stay robust to SDK reshuffles instead
    of importing concrete openai exception classes.
    """
    exc_name = type(e).__name__
    if exc_name in _RETRIABLE_EXC_NAMES:
        return True
    status_code = getattr(e, "status_code", None)
    return status_code is not None and status_code >= 500


def _action_reconstruct(uid, tier_config, body):
    """Default action: disfluency reconstruction (the original endpoint)."""
    raw = body.get("raw", "").strip()
    if not raw:
        return (json.dumps({"error": "Missing 'raw' field"}), 400, CORS_HEADERS)

    requested_layer = body.get("layer", 2)
    if requested_layer > tier_config["max_layer"]:
        return (json.dumps({
            "error": f"Layer {requested_layer} requires Pro tier",
            "max_layer": tier_config["max_layer"],
            "tier": tier_config["name"],
        }), 403, CORS_HEADERS)

    ok, remaining, rate_err = check_rate_limit(uid, tier_config, requested_layer)
    if not ok:
        return rate_err

    t_call = time.time()
    try:
        result = reconstruct_intent(
            raw_text=raw,
            tone=body.get("tone", "casual"),
            layer=requested_layer,
            profile=body.get("profile"),
            situation=body.get("situation", "default"),
            mode=body.get("mode", "SAFE"),
            whisper_low_conf=body.get("whisper_low_conf"),
            whisper_disagreements=body.get("whisper_disagreements"),
            speech_severity_mod=body.get("speech_severity_mod", 0.0),
            paralinguistic_events=body.get("paralinguistic_events"),
            prosodic_context=body.get("prosodic_context"),
            language_code=body.get("language_code", "en"),
            preceding_context=body.get("preceding_context"),
            script_prep_context=body.get("script_prep"),
            compression_ratio_note=body.get("compression_ratio_note"),
            previous_outputs=body.get("previous_outputs"),
            prior_rejections=body.get("rejection_history"),
            style_examples=body.get("style_examples"),
            window_title=body.get("audience_package"),
        )
    except Exception as e:
        latency_ms = round((time.time() - t_call) * 1000)
        retriable = _classify_exception(e)
        exc_name = type(e).__name__
        _emit(
            logging.ERROR,
            event="reconstruct_failed",
            uid=uid, layer=requested_layer, latency_ms=latency_ms,
            exception=exc_name, error=str(e)[:300], retriable=retriable,
        )
        return (json.dumps({
            "error": f"Reconstruction failed: {exc_name}",
            "retriable": retriable,
            "tier": tier_config["name"],
        }), 503 if retriable else 500, CORS_HEADERS)

    latency_ms = round((time.time() - t_call) * 1000)
    _emit(
        logging.INFO,
        event="reconstruct_ok",
        uid=uid, layer=requested_layer, latency_ms=latency_ms,
        model=result.get("model", "n/a"),
        mode=result.get("mode"),
    )
    result["tier"] = tier_config["name"]
    result["remaining"] = remaining
    return (json.dumps(result), 200, _JSON_CORS)


_ACTION_HANDLERS = {
    "sync_profile":     _action_sync_profile,
    "export_data":      _action_export_data,
    "delete_data":      _action_delete_data,
    "command":          _action_command,
    "complete_partial": _action_complete_partial,
    "transcribe_audio": _action_transcribe_audio,
    "learn_profile":    _action_learn_profile,
    "billing_status":   _action_billing_status,
    "verify_purchase":  _action_verify_purchase,
}

_ENTITLEMENT_FREE_ACTIONS = {
    "billing_status", "verify_purchase", "export_data", "delete_data", "sync_profile",
}


@functions_framework.http
def handle(request):
    """HTTP Cloud Function entry point — preflight + auth + dispatch."""
    if request.method == "OPTIONS":
        return ("", 204, CORS_HEADERS)
    if request.method != "POST":
        return (json.dumps({"error": "POST required"}), 405, CORS_HEADERS)

    identity, err = verify_token(request)
    if err:
        return err

    uid = identity["uid"]
    reviewer = reviewer_email_is_allowed(
        identity.get("email"),
        identity.get("email_verified"),
        os.environ.get("WIM_REVIEWER_EMAILS", ""),
    )
    tier_name = get_user_tier(uid, reviewer=reviewer)
    tier_config = TIERS.get(tier_name, TIERS["invite"])

    if (request.content_type or "").lower().startswith("multipart/form-data"):
        # Raw envelope (2026-09-06): the recording rides as the `file` part,
        # every other field is a plain form string. Only transcribe_audio
        # uses it; the bytes travel inside the body under a private key so
        # the action handlers keep one signature.
        try:
            body = {k: v for k, v in request.form.items()}
            upload = request.files.get("file")
            body["_audio_bytes"] = upload.read() if upload is not None else b""
        except Exception:
            return (json.dumps({"error": "Invalid multipart body"}), 400, CORS_HEADERS)
    else:
        try:
            body = request.get_json(silent=True) or {}
        except Exception:
            return (json.dumps({"error": "Invalid JSON"}), 400, CORS_HEADERS)

    action = body.get("action")
    if tier_name not in ("basic", "pro") and action not in _ENTITLEMENT_FREE_ACTIONS:
        return (json.dumps({
            "error": "An active WiM Cloud subscription is required",
            "error_code": "billing_required",
            "product_id": BILLING_PRODUCT_ID,
        }), 403, _JSON_CORS)

    action_handler = _ACTION_HANDLERS.get(action, _action_reconstruct)
    return action_handler(uid, tier_config, body)


def billing_event(event, context=None):
    """Apply Google Play RTDN subscription changes to Firestore.

    Pub/Sub authenticates the Google Play publisher; the purchase token in the
    event is then checked against Google Play's subscriptionsv2 source of truth.
    Only a token already bound to a WiM user can change that user's entitlement.
    """
    try:
        message_id, payload = decode_pubsub_cloud_event(event)
    except BillingEventError as exc:
        _emit(logging.WARNING, event="billing_rtdn_invalid", error=str(exc))
        return

    if payload.get("packageName") != BILLING_PACKAGE_NAME:
        _emit(logging.WARNING, event="billing_rtdn_wrong_package",
              message_id=message_id)
        return
    notification = payload.get("subscriptionNotification")
    if not isinstance(notification, dict):
        # Play Console sends a testNotification before the subscription exists.
        _emit(logging.INFO, event="billing_rtdn_ignored", message_id=message_id)
        return
    purchase_token = (notification.get("purchaseToken") or "").strip()
    if not purchase_token:
        _emit(logging.WARNING, event="billing_rtdn_missing_token",
              message_id=message_id)
        return

    hashed_token = token_hash(purchase_token)
    token_ref = db.collection("wim_subscription_tokens").document(hashed_token)
    token_snapshot = token_ref.get()
    if not token_snapshot.exists:
        # A purchase callback can race the RTDN. The app will bind and verify it
        # on the next foreground; never guess a user from an unbound token.
        _emit(logging.INFO, event="billing_rtdn_unbound", message_id=message_id)
        return
    uid = token_snapshot.to_dict().get("uid")
    if not uid:
        _emit(logging.WARNING, event="billing_rtdn_missing_uid",
              message_id=message_id)
        return

    try:
        purchase, _ = fetch_subscription_with_google(
            purchase_token, BILLING_PRODUCT_ID)
    except BillingVerificationError as exc:
        # Transient Publisher API errors must raise so Pub/Sub retries. A token
        # that is permanently gone (410) is safe to revoke if it is still the
        # user's current bound token.
        if exc.status != 410:
            _emit(logging.ERROR, event="billing_rtdn_lookup_failed",
                  message_id=message_id, status=exc.status)
            raise
        purchase = {
            "subscriptionState": "SUBSCRIPTION_STATE_EXPIRED",
            "_wim_expiry_ts": 0,
            "_wim_line_item": {},
        }

    entitled = subscription_is_entitled(purchase)
    state = purchase.get("subscriptionState")
    expiry_ts = float(purchase.get("_wim_expiry_ts", 0) or 0)
    line_item = purchase.get("_wim_line_item") or {}
    order_id = line_item.get("latestSuccessfulOrderId") or purchase.get("latestOrderId")
    user_ref = db.collection("wim_users").document(uid)

    @firestore.transactional
    def _apply(transaction):
        current_token = token_ref.get(transaction=transaction)
        user = user_ref.get(transaction=transaction)
        if not current_token.exists or current_token.to_dict().get("uid") != uid:
            return False
        user_data = user.to_dict() if user.exists else {}
        transaction.set(token_ref, {
            "subscription_state": state,
            "expiry_ts": expiry_ts,
            "order_id": order_id,
            "rtdn_message_id": message_id,
            "rtdn_updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)
        # An old token must never revoke a newer resubscription.
        if user_data.get("billing_token_hash") != hashed_token:
            return False
        update = {
            "billing_state": state,
            "billing_expiry_ts": expiry_ts,
            "billing_order_id": order_id,
            "billing_verified_at": firestore.SERVER_TIMESTAMP,
        }
        if entitled:
            update["tier"] = "basic"
        elif (user_data.get("tier") == "basic" and
              user_data.get("billing_product_id") == BILLING_PRODUCT_ID):
            update["tier"] = "invite"
        transaction.set(user_ref, update, merge=True)
        return True

    applied = _apply(db.transaction())
    _emit(logging.INFO, event="billing_rtdn_applied", message_id=message_id,
          uid=uid, state=state, entitled=entitled, applied=applied)
