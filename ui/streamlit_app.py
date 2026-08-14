"""Streamlit client.

Two panes: the conversation on the left, the agent's internals on the right.

The brief asks that an evaluator be able to see what the agent is doing, so the
right-hand panel is treated as a product surface rather than a debug view. It
renders the same SSE stream the answer arrives on, in the order events actually
occurred — not a summary assembled afterwards.

The client holds no authorization logic. It shows the role and access ceiling
the API reported, and it renders whatever it is sent. Every decision about what
this user may see was made server-side, and duplicating any of it here would
create a second, weaker copy of the rules.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)

DEMO_ACCOUNTS = [
    ("viewer", "viewer-demo-2026", "chat + search, internal documents"),
    ("analyst", "analyst-demo-2026", "+ analytics, MCP, confidential documents"),
    ("admin", "admin-demo-2026", "+ administrative tools, restricted documents"),
]

#: How each activity event is labelled in the panel. Keeping the mapping here,
#: rather than deriving a label from the event name, means the panel reads as
#: English rather than as an event log.
EVENT_LABELS = {
    "node.enter": ("▸", "entered"),
    "node.exit": ("▪", "finished"),
    "plan.update": ("☰", "planned"),
    "tool.call": ("⚙", "tool call"),
    "tool.result": ("⚙", "tool result"),
    "retrieval.stage": ("⌕", "retrieval"),
    "memory.read": ("◔", "memory read"),
    "memory.write": ("◕", "memory write"),
    "validation.result": ("✓", "validation"),
    "guard.result": ("⛨", "guard"),
    "recursion": ("↻", "recursion"),
    "degradation": ("⚠", "degraded"),
    "budget": ("◷", "budget"),
}


st.set_page_config(page_title="Atrium", page_icon="🏛", layout="wide")


def _init_state() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("principal", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("activity", [])


def login(user_id: str, password: str) -> tuple[bool, str]:
    try:
        response = httpx.post(
            f"{API_BASE}/auth/login",
            json={"user_id": user_id, "password": password},
            timeout=REQUEST_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return False, f"Could not reach the API at {API_BASE} ({type(exc).__name__})."

    if response.status_code != 200:
        detail = response.json().get("detail", "Sign-in failed.")
        return False, str(detail)

    body = response.json()
    st.session_state.token = body["access_token"]
    st.session_state.principal = body["principal"]
    return True, ""


def render_login() -> None:
    st.title("Atrium")
    st.caption("Internal knowledge assistant · Commercial Bank")

    left, right = st.columns([1, 1])
    with left, st.form("login"):
        user_id = st.text_input("User", value="analyst")
        password = st.text_input("Password", type="password", value="analyst-demo-2026")
        if st.form_submit_button("Sign in", use_container_width=True):
            ok, error = login(user_id, password)
            if ok:
                st.rerun()
            else:
                st.error(error)

    with right:
        st.markdown("**Demo accounts**")
        st.caption("Each role sees a different slice of the same corpus.")
        for name, secret, scope in DEMO_ACCOUNTS:
            st.markdown(f"`{name}` / `{secret}`  \n&nbsp;&nbsp;{scope}", unsafe_allow_html=True)


def render_sidebar() -> None:
    principal = st.session_state.principal
    with st.sidebar:
        st.markdown(f"### {principal['display_name']}")
        st.caption(f"{principal['role']} · sees up to **{principal['access_ceiling']}**")
        st.caption("Departments: " + ", ".join(principal["departments"]))

        try:
            limits = httpx.get(
                f"{API_BASE}/chat/limits",
                headers={"Authorization": f"Bearer {st.session_state.token}"},
                timeout=REQUEST_TIMEOUT,
            ).json()
            st.progress(
                min(1.0, limits["requests_remaining"] / max(limits["requests_capacity"], 1)),
                text=f"{limits['requests_remaining']:.0f} requests left",
            )
        except (httpx.HTTPError, KeyError, ValueError):
            # The meter is a nicety; failing to draw it must not block the chat.
            pass

        if st.button("New conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.activity = []
            st.session_state.thread_id = None
            st.rerun()

        if st.button("Sign out", use_container_width=True):
            for key in ("token", "principal", "messages", "activity", "thread_id"):
                st.session_state[key] = None if key in ("token", "principal") else []
            st.rerun()

        st.divider()
        st.caption(
            "Authorization is enforced server-side, in the retrieval filter. "
            "This client renders what it is sent and decides nothing."
        )


def _summarise(event_type: str, data: dict[str, Any]) -> str:
    """One line describing an event, written for a person."""
    if event_type == "retrieval.stage":
        stage = data.get("stage", "")
        count = data.get("count", 0)
        extra = ""
        if "alpha" in data:
            extra = f" · alpha {data['alpha']}"
        elif "namespaces" in data:
            extra = f" · {data['namespaces']} namespace(s)"
        elif "query" in data:
            extra = f" · “{str(data['query'])[:60]}”"
        return f"{stage}: {count}{extra}"

    if event_type == "guard.result":
        signals = ", ".join(data.get("signals", [])[:2])
        return f"{data.get('verdict')} ({data.get('score')})" + (f" · {signals}" if signals else "")

    if event_type == "validation.result":
        verdict = "passed" if data.get("passed") else "failed"
        return (
            f"{verdict} · {data.get('grounded_claims', 0)} grounded, "
            f"{data.get('ungrounded_claims', 0)} unsupported (attempt {data.get('attempt', 0) + 1})"
        )

    if event_type == "degradation":
        return f"{data.get('component')} unavailable → {data.get('fallback')}"

    if event_type == "plan.update":
        steps = data.get("steps", [])
        return " → ".join(f"{s['agent']}" for s in steps) or "no steps"

    if event_type == "tool.call":
        return f"{data.get('tool')} ({'allowed' if data.get('allowed') else 'denied'})"

    if event_type == "tool.result":
        return f"{data.get('tool')} · {data.get('duration_ms')}ms"

    if event_type == "budget":
        return f"{data.get('tool_calls')} tool calls, {data.get('supervisor_steps')} steps left"

    interesting = {k: v for k, v in data.items() if k not in {"stage"}}
    return ", ".join(f"{k}={v}" for k, v in list(interesting.items())[:3])


def render_activity(placeholder: Any) -> None:
    """Redraw the activity panel from the events collected so far.

    Writes into an emptied placeholder rather than appending to a container.
    Streamlit containers accumulate on every write, so re-rendering per event
    would stack a fresh copy of the whole list each time instead of updating it.
    """
    with placeholder.container():
        st.markdown("#### Agent activity")
        if not st.session_state.activity:
            st.caption("Ask a question to watch the agent work.")
            return

        for event in st.session_state.activity[-90:]:
            event_type = event.get("type", "")
            node = event.get("node", "")
            data = event.get("data", {})
            icon, label = EVENT_LABELS.get(event_type, ("·", event_type))
            depth = "&nbsp;" * (4 * int(event.get("depth", 0)))

            summary = _summarise(event_type, data)
            colour = (
                "#b45309"
                if event_type == "degradation"
                else "#0c6355"
                if event_type in {"validation.result", "guard.result"}
                else "#475569"
            )

            if event_type == "node.enter":
                st.markdown(
                    f"{depth}<span style='color:{colour}'><b>{icon} {node}</b></span>",
                    unsafe_allow_html=True,
                )
            elif summary:
                st.markdown(
                    f"{depth}<span style='color:{colour};font-size:0.85em'>"
                    f"{icon} {label} · {summary}</span>",
                    unsafe_allow_html=True,
                )


def stream_answer(question: str, activity_placeholder: Any) -> dict[str, Any]:
    """POST the question and consume the SSE stream, updating the panel live."""
    payload: dict[str, Any] = {"question": question}
    if st.session_state.thread_id:
        payload["thread_id"] = st.session_state.thread_id

    result: dict[str, Any] = {"text": "", "citations": [], "errors": []}

    with httpx.stream(
        "POST",
        f"{API_BASE}/chat/stream",
        json=payload,
        headers={"Authorization": f"Bearer {st.session_state.token}"},
        timeout=REQUEST_TIMEOUT,
    ) as response:
        if response.status_code != 200:
            response.read()
            try:
                problem = response.json()
                detail = problem.get("detail", "The request was rejected.")
                if response.status_code == 429:
                    retry = problem.get("retry_after_seconds", "a moment")
                    detail = f"{detail} Try again in {retry}s."
            except ValueError:
                detail = f"HTTP {response.status_code}"
            result["errors"].append(detail)
            return result

        event_type = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                raw = line.split(":", 1)[1].strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if event_type == "start":
                    st.session_state.thread_id = data.get("thread_id")
                elif event_type == "answer":
                    result.update(data)
                elif event_type == "error":
                    result["errors"].append(data.get("detail", "Unknown error"))
                elif event_type not in {"done", "start"}:
                    st.session_state.activity.append(data)
                    render_activity(activity_placeholder)

    return result


def render_chat() -> None:
    chat_column, activity_column = st.columns([3, 2], gap="large")

    with chat_column:
        st.markdown("### Ask about internal documentation")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                for citation in message.get("citations", []):
                    st.caption(
                        f"[{citation['chunk_id']}] {citation['title']} · "
                        f"{citation['heading_path']} · {citation['access_level']}"
                    )

    activity_placeholder = activity_column.empty()
    render_activity(activity_placeholder)

    question = st.chat_input("e.g. what caused the recurring payment failures?")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.activity = []

    with chat_column:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"), st.spinner("Working…"):
            result = stream_answer(question, activity_placeholder)

            if result["errors"] and not result["text"]:
                st.error(result["errors"][0])
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠ {result['errors'][0]}"}
                )
                return

            st.markdown(result["text"])
            if result.get("degraded"):
                st.warning(
                    "The primary search index was unavailable; this answer came "
                    "from a keyword-only fallback and may be less complete."
                )
            for citation in result.get("citations", []):
                st.caption(
                    f"[{citation['chunk_id']}] {citation['title']} · "
                    f"{citation['heading_path']} · {citation['access_level']}"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["text"],
            "citations": result.get("citations", []),
        }
    )


def main() -> None:
    _init_state()
    if not st.session_state.token:
        render_login()
        return
    render_sidebar()
    render_chat()


main()
