"""Unit tests for the deterministic (no-LLM) reconcile decision function.

These test the pure decision rule `Memory._decide` with prepared candidate
lists carrying representative cosine scores — no network, no store. They pin
the behavior on the calibrated benchmark cases.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memlite.core import Memory, _is_retraction, _shared_bigrams

D = Memory._decide  # static pure function


def _cand(text, cos, mid="id-1"):
    return {"id": mid, "memory": text, "score": cos}


def test_retraction_detection():
    assert _is_retraction("forget that I said I love pineapple on pizza")
    assert _is_retraction("remove the thing about coffee")
    assert _is_retraction("I never liked spinach")
    assert _is_retraction("no longer prefer dark mode")
    assert not _is_retraction("don't forget to buy milk")  # positive intent
    assert not _is_retraction("my favorite color is teal")


def test_shared_bigrams():
    # attribute key phrase survives a value change -> shared bigram
    assert _shared_bigrams("my favorite color is teal",
                           "user's favorite color is teal") >= {"favorite color"}
    # same entity, different fact -> no shared content bigram
    assert _shared_bigrams("user went hiking with max",
                           "user has a dog named max") == set()


def test_exact_reword_updates():
    # cos 0.878, jacc ~0.75 -> UPDATE
    op = D("my favorite color is teal", None, [_cand("user's favorite color is teal", 0.878)])
    assert op["event"] == "UPDATE", op


def test_topic_change_updates():
    # "changed to magenta" vs "is teal": cos 0.728, jacc ~0.40 -> UPDATE
    op = D("actually my favorite color changed to magenta", None,
           [_cand("user's favorite color is teal", 0.728)])
    assert op["event"] == "UPDATE", op


def test_same_entity_different_fact_adds():
    # "user went hiking with Max" vs "user has a dog named Max":
    # cos 0.761 (high!) but jacc ~0.2 (shared entity, different event) -> must ADD
    op = D("User went hiking with Max", None,
           [_cand("User has a dog named Max", 0.761)])
    assert op["event"] == "ADD", op


def test_unrelated_adds():
    op = D("User prefers dark mode in the editor", None,
           [_cand("I love pineapple on pizza", 0.384)])
    assert op["event"] == "ADD", op


def test_retraction_deletes_close_match():
    # "forget pineapple" vs existing pineapple fact: cos 0.851 -> DELETE
    op = D("Forget that I said I love pineapple on pizza", None,
           [_cand("User loves pineapple on pizza", 0.851)])
    assert op["event"] == "DELETE", op


def test_retraction_without_match_adds():
    # retraction intent but no close existing memory -> nothing to retract -> ADD
    op = D("forget about the skydiving plan", None, [_cand("User loves hiking", 0.45)])
    assert op["event"] == "ADD", op


def test_empty_existing_adds():
    assert D("brand new fact", None, [])["event"] == "ADD"


def test_low_overlap_high_cos_refines_is_add():
    # high cosine but low whitetoken overlap (same noun "honda", different car)
    # -> conservative ADD (keep both) per the data-retention bias.
    op = D("My car is a black Honda Accord", None,
           [_cand("I drive a red Honda Civic", 0.708)])
    assert op["event"] == "ADD", op


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nALL DETERMINISTIC RECONCILE UNIT TESTS PASSED")
