"""docchat's face: a chat page in front of the API. Lesson 5.

    python -m app.serve              # the API, in one terminal
    streamlit run ui/app.py          # this, in another: http://localhost:8501

Streamlit runs this file from top to bottom every time the user does anything:
types a question, clicks a button, picks a file. Every line below runs again on
each of those reruns, and every local variable starts from scratch. The only
thing that survives is `st.session_state`, a dict kept per browser tab. Read the
file with that in mind and most of it explains itself:

    session_state     who the user is, which conversation is open, its messages
    sidebar           health, usage, conversations, documents: fetched on every rerun
    the transcript    redrawn from session_state on every rerun
    chat_input        returns the question only on the rerun that submitted it

The page knows nothing about retrieval, prompts or Postgres. It holds a
conversation ID and a list of messages, and calls `ui/client.py` for everything
else. The API decides what is remembered (Lesson 4); this file only draws it.
"""

from __future__ import annotations

import re
import time

import streamlit as st

from ui.client import API_URL, APIError, Client, log, setup_logging

st.set_page_config(page_title="docchat", page_icon=":material/description:")

# ---- what survives a rerun ---------------------------------------------------
# setdefault writes only on a tab's first run. On every later rerun these keys
# already hold what the user did, and this loop leaves them alone.
for key, value in {"user": "ana", "conversation_id": None, "messages": [],
                   "uploading": None, "upload_result": None, "uploader": 0, "runs": 0}.items():
    st.session_state.setdefault(key, value)

# One line per run in the terminal running `streamlit run`. Click anything and a new
# one appears: that is Section 2's rerun, made visible. A fragment rerunning on its
# timer does not run this line, so its polls show up without a run before them.
setup_logging()
st.session_state.runs += 1
log.info("run %d: user %r, conversation %s, %d messages on screen", st.session_state.runs,
         st.session_state.user, st.session_state.conversation_id, len(st.session_state.messages))


def api() -> Client:
    return Client(st.session_state.user.strip() or "anonymous")


def new_conversation() -> None:
    st.session_state.conversation_id, st.session_state.messages = None, []


def open_conversation(conversation_id: int) -> None:
    """Callbacks run before the rerun they cause, so the page draws the new state."""
    try:
        conv = api().conversation(conversation_id)
    except APIError as e:
        st.toast(f"Could not open conversation {conversation_id}: {e.detail}")
        return
    st.session_state.conversation_id = conv["id"]
    st.session_state.messages = conv["messages"]


# ---- text a renderer will not misread ----------------------------------------
CITE = re.compile(r"\[(\d+)\]")


def answer_markdown(text: str, citations: list[dict]) -> str:
    """The answer with its [n] markers drawn as badges.

    Two dollar amounts on one line are LaTeX to Streamlit's markdown, and annual
    reports are full of them, so every $ is escaped. A marker the model wrote
    with no matching block (it said [7] and got five) is drawn grey: the API
    dropped it from the citations, and the reader should see that it is empty."""
    known = {c["marker"] for c in citations}
    parts = CITE.split(text.replace("$", r"\$"))
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append(part)
        else:
            out.append(f":blue-badge[{part}]" if int(part) in known else f":gray-badge[{part}?]")
    return "".join(out)


# ---- the sources panel -------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def passage(chunk_id: str) -> dict | None:
    """Cached for five minutes: a rerun redraws every answer's sources, and a
    passage does not change between reruns. It can change when a document is
    uploaded again, which is why the cache expires."""
    return Client("anonymous").chunk(chunk_id)


def sources_panel(citations: list[dict]) -> None:
    """Where each marker points: document, page, score, and the passage itself."""
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in sorted(citations, key=lambda c: c["marker"]):
            p = passage(c["chunk_id"])
            title = p["title"] if p else c["doc"]
            st.markdown(f":blue-badge[{c['marker']}] **{title}**, page {c['page']} "
                        f"&nbsp; :gray[relevance {c['score']:.2f}]")
            if p is None:
                st.caption("This passage is no longer in the corpus: its document "
                           "was deleted or replaced since the answer was written.")
            else:
                st.text(p["text"], width="stretch")


# ---- one message, in whichever state it is in --------------------------------
def show(m: dict) -> None:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"].replace("$", r"\$"))
            return
        if m.get("content") and not m.get("refused"):
            st.markdown(answer_markdown(m["content"], m.get("citations", [])))
        if m.get("refused"):
            st.info("No answer: the documents do not cover this."
                    + (f"  \n:gray[{m['reason']}]" if m.get("reason") else ""),
                    icon=":material/search_off:")
        if err := m.get("error"):
            if err["status"] == 429:
                st.warning(f"{err['detail']}. Try again in {err['retry_after'] / 3600:.1f} hours.",
                           icon=":material/hourglass_top:")
            else:
                st.error(err["detail"], icon=":material/error:")
        sources_panel(m.get("citations", []))
        if m.get("model"):
            meta = [m["model"], f"{m['n_in']:,} in / {m['n_out']:,} out", f"${m['usd']:.5f}"]
            if m.get("seconds") is not None:
                meta.append(f"{m['seconds']:.1f}s")
            if m.get("cached"):
                meta.append("from the cache")
            st.caption(" · ".join(meta).replace("$", r"\$"))


def reply_to(question: str) -> dict:
    """Stream one answer onto the page and return it as a message to keep.

    `st.write_stream` draws each `delta` as it arrives. The other events fill in
    the message around it: citations and refusal, usage, and finally `done` with
    the conversation ID, which is what makes the next question a follow-up."""
    m = {"role": "assistant", "content": "", "citations": [], "refused": False}

    def deltas():
        for event, data in api().stream(question, st.session_state.conversation_id):
            if event == "delta":
                yield data["text"]
            elif event == "citations":
                m.update(data)
            elif event == "usage":
                m.update(data)
            elif event == "done":
                m.update(seconds=data["seconds"], cached=data["cached"])
                st.session_state.conversation_id = data["conversation_id"]
            elif event == "error":
                m["error"] = {"status": None, "detail": f"The answer stopped: {data['message']}"}

    try:
        m["content"] = st.write_stream(deltas()) or ""
    except APIError as e:
        m["error"] = {"status": e.status, "detail": e.detail, "retry_after": e.retry_after or 0}
        if e.status == 404:             # the conversation is gone, or is not ours
            st.session_state.conversation_id = None
            m["error"]["detail"] += ". Your next question starts a new conversation."
    return m


# ---- the upload status -------------------------------------------------------
def upload_status() -> None:
    """The job pattern from Lesson 3, made visible: pending, then ready or failed.

    Polling has to happen somewhere, and a rerun of the whole page every second
    would refetch the sidebar and redraw the transcript each time. A fragment is
    a function Streamlit can rerun on its own, on a timer, while the rest of the
    page stays as it is. The timer is set only while something is uploading."""
    job = st.session_state.uploading
    if job is not None:
        d = api().document(job["id"])
        seconds = time.time() - job["started"]
        if d["status"] == "pending":
            st.info(f"Ingesting **{d['title']}**: parsing, chunking, embedding. "
                    f"{seconds:.0f}s so far.", icon=":material/hourglass_top:")
            return
        st.session_state.uploading, st.session_state.upload_result = None, {**d, "seconds": seconds}
        st.rerun()                      # the whole page: the document list has changed
    if d := st.session_state.upload_result:
        if d["status"] == "ready":
            st.success(f"**{d['title']}** is ready: {d['pages']} pages, {d['chunks']} chunks, "
                       f"{d['seconds']:.0f}s.", icon=":material/check_circle:")
        else:
            st.error(f"**{d['title']}** failed: {d['error']}", icon=":material/error:")


def documents_panel() -> None:
    st.subheader("Documents")
    f = st.file_uploader("Add a PDF", type="pdf", key=f"uploader-{st.session_state.uploader}")
    if f is not None:
        # The uploader keeps its file across reruns, so this branch would upload it
        # again on every click anywhere on the page. A new key makes a new, empty widget.
        st.session_state.uploader += 1
        try:
            d = api().upload(f.name, f.getvalue())
            st.session_state.uploading = {"id": d["id"], "started": time.time()}
            st.session_state.upload_result = None
        except APIError as e:
            st.session_state.upload_result = {"status": "failed", "title": f.name, "error": e.detail}
        st.rerun()
    st.fragment(run_every=1.0 if st.session_state.uploading else None)(upload_status)()
    icon = {"ready": ":material/check_circle:", "pending": ":material/hourglass_top:",
            "failed": ":material/error:"}
    for d in api().documents():
        st.markdown(f"{icon[d['status']]} {d['title']} :gray[· {d['pages']} pages]")


# ---- the page, top to bottom -------------------------------------------------
with st.sidebar:
    st.title("docchat")
    try:
        h = api().health()
    except APIError as e:
        st.error(f"{e.detail}. Start it with `python -m app.serve`.", icon=":material/cloud_off:")
        st.stop()                       # nothing below can work without the API
    st.caption(f"{API_URL} · {h['model']} · {h['chunks']:,} chunks")

    st.text_input("User", key="user", on_change=new_conversation,
                  help="Sent as X-User-Id. A label, not a login: Lesson 4, Section 3.")
    u = api().usage()
    st.progress(min(1.0, u["used"] / u["limit"]),
                text=f"{u['used']:,} of {u['limit']:,} tokens in {u['window_hours']} hours")

    st.button("New conversation", icon=":material/add:", on_click=new_conversation,
              width="stretch")
    for c in api().conversations():
        st.button(c["title"] or f"Conversation {c['id']}", key=f"conv-{c['id']}",
                  on_click=open_conversation, args=(c["id"],), width="stretch",
                  type="primary" if c["id"] == st.session_state.conversation_id else "tertiary")

    documents_panel()

st.header("Ask the annual reports")
if not st.session_state.messages:
    st.caption("Every answer cites the pages it came from. Open **Sources** under an answer "
               "to read the passages, and check the answer against them.")
for m in st.session_state.messages:
    show(m)

if question := st.chat_input("Ask a question about the documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    show(st.session_state.messages[-1])
    with st.chat_message("assistant"):
        m = reply_to(question)
    st.session_state.messages.append(m)
    # The sidebar was drawn before this answer existed: its usage bar and its list
    # of conversations are one turn out of date. Rerun, and everything is redrawn
    # from session_state and the API, the new answer with its sources included.
    st.rerun()
