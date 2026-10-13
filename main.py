import argparse
import json
import re
import textwrap
from typing import Optional

import sys

from openai import APIError, AuthenticationError, OpenAI, OpenAIError, RateLimitError

import prompts

"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

1. A better evaluation harness: a set of 30+ varied requests (including unsafe ones) scored by a
   separate judge prompt, so prompt changes can be compared on numbers rather than by feel. The
   current judge also scores its own revisions, which biases scores upward.
2. Reading the story out loud with a text-to-speech voice, and a "continue tomorrow night" mode
   that saves characters so a child can get a new episode with the same friends.
3. Streaming the final story to the terminal, and running the judge on a different temperature
   or with a few calibration examples to make its scores more consistent.
"""

MODEL = "gpt-3.5-turbo"  # fixed by the assignment, do not change
MAX_REVISIONS = 2        # judge -> revise rounds before we stop
# Pass rule. GPT-3.5's scores wobble by about +-1 between runs, so "every score >= 8" was mostly
# noise. Instead we require a high average and no criterion that is clearly weak.
PASS_AVG = 8.5
PASS_MIN = 7
MAX_FIXES = 3            # a few targeted fixes per round beat a long list the reviser half-applies
# Stories are read aloud, and children understand speech about two grades above what they can
# read themselves, so grade 6 text is fine for ages 5-10 as listeners.
MAX_READING_GRADE = 6.0
MIN_LENGTH_RATIO = 0.75  # below this fraction of the target length, add an automatic "expand" fix

_client = None


def call_model(prompt: str, system: Optional[str] = None, max_tokens: int = 3000,
               temperature: float = 0.1, json_mode: bool = False) -> str:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **extra,
    )
    return resp.choices[0].message.content or ""


def call_json(prompt: str, system: str, temperature: float = 0.0, max_tokens: int = 1000) -> dict:
    """Call the model in JSON mode and parse the result, retrying once on bad JSON."""
    raw = ""
    for _ in range(2):
        raw = call_model(prompt, system, max_tokens, temperature, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
    raise ValueError(f"Model did not return valid JSON: {raw[:200]}")


# ---------------------------------------------------------------------------
# Reading level: an objective signal the judge can't hallucinate
# ---------------------------------------------------------------------------

def count_syllables(word: str) -> int:
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1  # silent e
    return max(1, count)


def story_body(story: str) -> str:
    """The story without its title line, so the title doesn't skew length or reading level."""
    parts = story.split("\n", 1)
    return parts[1] if len(parts) == 2 else story


def split_sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])[\"')\s]+", text) if s.strip()]


def reading_grade(text: str) -> float:
    """Flesch-Kincaid grade level."""
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    sentences = max(1, len(split_sentences(text)))
    syllables = sum(count_syllables(w) for w in words)
    return 0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59


def word_count(text: str) -> int:
    return len(text.split())


def code_checks(brief: dict, story: str) -> list:
    """Objective checks done in Python. Returns specific fixes the judge can't hallucinate."""
    body = story_body(story)
    fixes = []
    grade = reading_grade(body)
    if grade > MAX_READING_GRADE:
        # Generic "use simpler words" didn't work, so point at the exact sentences to split.
        long_ones = [s for s in split_sentences(body) if len(s.split()) >= 14]
        longest = sorted(long_ones, key=lambda s: len(s.split()), reverse=True)[:3]
        quoted = " | ".join(f'"{s}"' for s in longest)
        target = f"Split these long sentences: {quoted}. " if longest else ""
        fixes.append(f"Reading level is grade {grade:.1f}, too hard. {target}"
                     "Swap long words for short everyday ones (e.g. 'enormous' -> 'huge').")
    words = word_count(body)
    if words < brief["words"] * MIN_LENGTH_RATIO:
        fixes.append(f"The story is only {words} words; it should be about {brief['words']}. "
                     "Expand the middle: add a second attempt to solve the problem, more dialogue, "
                     "and sensory details. Do not just add sentences to the ending.")
    return fixes


def clean_story(story: str) -> str:
    """Remove markdown separators/headings GPT-3.5 sometimes adds."""
    lines = [l for l in story.strip().split("\n") if not re.fullmatch(r"\s*([-*_=]{3,})\s*", l)]
    text = "\n".join(l.lstrip("#").strip() if l.startswith("#") else l for l in lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def classify(request: str) -> dict:
    categories = "\n".join(f"- {k}: {v}" for k, v in prompts.CATEGORIES.items())
    brief = call_json(
        prompts.CLASSIFIER_PROMPT.format(request=request, categories=categories),
        prompts.CLASSIFIER_SYSTEM,
    )
    # Fall back to safe defaults if the model returns something unexpected.
    if brief.get("category") not in prompts.CATEGORIES:
        brief["category"] = "adventure"
    if brief.get("length") not in prompts.LENGTH_WORDS:
        brief["length"] = "medium"
    brief.setdefault("story_request", request)
    brief.setdefault("safe", True)
    brief["words"] = prompts.LENGTH_WORDS[brief["length"]]
    brief["category_guide"] = prompts.CATEGORY_GUIDES[brief["category"]]
    return brief


def plan(brief: dict) -> dict:
    outline = call_json(
        prompts.PLANNER_PROMPT.format(**brief),
        prompts.PLANNER_SYSTEM,
        temperature=0.7,  # some creativity in the plot, but structured output
    )
    # GPT-3.5 sometimes puts stray items (like "lesson") into the beats list; keep only real beats.
    beats = [b for b in outline.get("beats", []) if isinstance(b, str) and len(b.split()) > 3]
    outline["beats"] = beats[:5]
    outline.setdefault("title", "A Bedtime Story")
    outline.setdefault("characters", [])
    return outline


def outline_to_notes(outline: dict) -> str:
    """Render the outline as author's notes rather than raw JSON.

    With raw JSON, GPT-3.5 copied each field into the story verbatim (one paragraph per beat,
    separated by ---). Notes phrased as plans are much less likely to be pasted in directly.
    """
    chars = "\n".join(f"- {c.get('name', '?')}: {c.get('description', '')}"
                      for c in outline["characters"] if isinstance(c, dict))
    beats = "\n".join(f"{i}. {re.sub(r'^[A-Za-z ]+ - ', '', b)}"
                      for i, b in enumerate(outline["beats"], 1))
    return (f"Title: {outline['title']}\n"
            f"Characters:\n{chars}\n"
            f"Where it happens: {outline.get('setting', '')}\n"
            f"Refrain to repeat 2-3 times: {outline.get('refrain', '')}\n"
            f"What happens (each point becomes several paragraphs):\n{beats}\n"
            f"Lesson to show, never state: {outline.get('lesson', '')}")


def write_story(brief: dict, outline: dict) -> str:
    return clean_story(call_model(
        prompts.STORYTELLER_PROMPT.format(
            notes=outline_to_notes(outline),
            paragraphs=prompts.LENGTH_PARAGRAPHS[brief["length"]],
            **brief,
        ),
        prompts.STORYTELLER_SYSTEM,
        temperature=0.8,  # creative writing benefits from more variety
    ))


def format_fix(fix) -> str:
    """Judge fixes are {criterion, quote, change} objects; turn them into one instruction."""
    if isinstance(fix, dict):
        quote = fix.get("quote", "").strip()
        change = fix.get("change", "").strip()
        where = f'Where it says "{quote}": ' if quote else ""
        return f"[{fix.get('criterion', 'general')}] {where}{change}"
    return str(fix)


def judge(brief: dict, story: str) -> dict:
    body = story_body(story)
    grade = reading_grade(body)
    rubric = "\n".join(f"- {k}: {v}" for k, v in prompts.RUBRIC.items())
    result = call_json(
        prompts.JUDGE_PROMPT.format(
            story=story, rubric=rubric, grade=grade, max_grade=MAX_READING_GRADE,
            actual_words=word_count(body), max_fixes=MAX_FIXES, **brief
        ),
        prompts.JUDGE_SYSTEM,
        temperature=0.0,  # consistent, repeatable grading
    )
    # Normalise: every rubric key gets an int score, missing ones count as failing.
    scores = result.get("scores", {})
    result["scores"] = {k: int(scores.get(k, 0) or 0) for k in prompts.RUBRIC}
    result.setdefault("strengths", [])
    # Code-measured problems go first: they are objective and the most reliable fixes.
    llm_fixes = [format_fix(f) for f in result.get("fixes", [])][:MAX_FIXES]
    result["fixes"] = code_checks(brief, story) + llm_fixes
    result["grade"] = grade
    result["min_score"] = min(result["scores"].values())
    result["avg_score"] = sum(result["scores"].values()) / len(result["scores"])
    # The pass decision is made in code, not by the judge, so it is predictable.
    result["passed"] = (result["avg_score"] >= PASS_AVG and result["min_score"] >= PASS_MIN
                        and grade <= MAX_READING_GRADE)
    return result


def revise(brief: dict, story: str, instructions: list, strengths: list) -> str:
    return clean_story(call_model(
        prompts.REVISER_PROMPT.format(
            story=story,
            instructions="\n".join(f"{n}. {i}" for n, i in enumerate(instructions, 1)),
            strengths="\n".join(f"- {s}" for s in strengths) or "- (none listed)",
            current_words=word_count(story_body(story)),
            **brief,
        ),
        prompts.STORYTELLER_SYSTEM,
        temperature=0.5,  # lower than drafting: we want targeted edits, not a new story
    ))


def rank(report: dict) -> tuple:
    """Order reports: passing beats failing, then by weakest score, then by average."""
    return (report["passed"], report["min_score"], report["avg_score"])


def judge_and_improve(brief: dict, story: str, verbose: bool = False):
    """Judge the story and revise it until it passes or we run out of rounds.

    Keeps the best-scoring version, so a bad revision can never make the result worse.
    Returns (best_story, best_report, history).
    """
    report = judge(brief, story)
    history = [report]
    best_story, best_report = story, report
    log_round("first draft", report, verbose)

    for round_no in range(1, MAX_REVISIONS + 1):
        if best_report["passed"]:
            break
        story = revise(brief, best_story, best_report["fixes"], best_report["strengths"])
        report = judge(brief, story)
        history.append(report)
        log_round(f"revision {round_no}", report, verbose)
        if rank(report) > rank(best_report):
            best_story, best_report = story, report
    return best_story, best_report, history


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def status(msg: str):
    print(f"  · {msg}", flush=True)


def log_round(label: str, report: dict, verbose: bool):
    verdict = "pass" if report["passed"] else "needs work"
    status(f"Judge on {label}: min {report['min_score']}/10, avg {report['avg_score']:.1f}, "
           f"reading grade {report['grade']:.1f} -> {verdict}")
    if verbose:
        for k, v in report["scores"].items():
            print(f"        {k:<16} {v}")
        for fix in report["fixes"]:
            print(f"        fix: {fix}")


def print_story(story: str):
    print("\n" + "=" * 70 + "\n")
    for para in story.split("\n"):
        print(textwrap.fill(para, 70) if para.strip() else "")
    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main flows
# ---------------------------------------------------------------------------

def tell_story(request: str, verbose: bool = False) -> None:
    status("Reading your request...")
    brief = classify(request)
    if not brief["safe"] and brief.get("note_to_user"):
        print(f"\n{brief['note_to_user']}\n")
    status(f"Category: {brief['category']}, length: {brief['length']}")

    status("Planning the story arc...")
    outline = plan(brief)
    if verbose:
        print(json.dumps(outline, indent=2))

    status("Writing the first draft...")
    story = write_story(brief, outline)
    story, report, _ = judge_and_improve(brief, story, verbose)
    print_story(story)

    # Feedback loop: the user's request goes through the same reviser and judge.
    while True:
        feedback = input("Want any changes? Describe them, or press Enter to finish: ").strip()
        if not feedback:
            print("Sweet dreams!")
            return
        status("Rewriting with your changes...")
        story = revise(brief, story, [feedback, prompts.USER_FEEDBACK_SAFETY_NOTE], report["strengths"])
        report = judge(brief, story)
        log_round("your changes", report, verbose)
        if not report["passed"]:
            # One judge-guided polish pass on top of the user's change. We don't keep the
            # pre-feedback version here, since the user explicitly asked for a change.
            fixes = [f"Keep this change the reader asked for: {feedback}"] + report["fixes"]
            story = revise(brief, story, fixes, report["strengths"])
            report = judge(brief, story)
            log_round("polished version", report, verbose)
        print_story(story)


EVAL_REQUESTS = [
    "A story about a girl named Alice and her best friend Bob, who happens to be a cat.",
    "A dragon who is afraid of the dark",
    "Tell me a scary story about zombies attacking a town",
    "A funny story about a pizza that wants to be a pet",
    "Something calm about the moon and the ocean to help me fall asleep",
]


def run_eval() -> None:
    """Show how much the judge loop improves first drafts across a set of requests."""
    rows = []
    for req in EVAL_REQUESTS:
        print(f"\n> {req}")
        brief = classify(req)
        story = write_story(brief, plan(brief))
        final, best, history = judge_and_improve(brief, story)
        rows.append((req, brief, history[0], best, len(history) - 1, word_count(story_body(final))))

    line = "-" * 110
    print("\n" + line)
    print(f"{'request':<40} {'category':<15} {'safe':<5} {'draft min/avg':>13} "
          f"{'final min/avg':>13} {'rounds':>6} {'words':>6} {'pass':>5}")
    print(line)
    for req, brief, first, best, rounds, words in rows:
        print(f"{req[:39]:<40} {brief['category']:<15} {'yes' if brief['safe'] else 'no':<5} "
              f"{first['min_score']:>5} / {first['avg_score']:<5.1f} "
              f"{best['min_score']:>5} / {best['avg_score']:<5.1f} {rounds:>6} {words:>6} "
              f"{'yes' if best['passed'] else 'no':>5}")
    print(line)
    gain = sum(b["avg_score"] - f["avg_score"] for _, _, f, b, _, _ in rows) / len(rows)
    passed = sum(b["passed"] for _, _, _, b, _, _ in rows)
    print(f"Average judge score gain from the revision loop: {gain:+.2f}. Passed: {passed}/{len(rows)}.")


def main():
    parser = argparse.ArgumentParser(description="Bedtime story generator for ages 5-10")
    parser.add_argument("request", nargs="*", help="story request (asked interactively if omitted)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show outline, scores and fixes")
    parser.add_argument("--eval", action="store_true", help="run the judge-loop evaluation set")
    args = parser.parse_args()

    if args.eval:
        run_eval()
        return
    user_input = " ".join(args.request) or input("What kind of story do you want to hear? ")
    tell_story(user_input, args.verbose)


if __name__ == "__main__":
    try:
        main()
    except AuthenticationError:
        sys.exit("\nOpenAI rejected the API key. Check that OPENAI_API_KEY is set correctly.")
    except RateLimitError as e:
        sys.exit(f"\nOpenAI rate limit or quota error (check your billing/credits): {e.message}")
    except (APIError, OpenAIError) as e:
        sys.exit(f"\nOpenAI error: {e}")
    except KeyboardInterrupt:
        sys.exit("\nGoodnight!")
