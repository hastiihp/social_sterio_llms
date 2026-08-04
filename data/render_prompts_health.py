"""
Generates the health-conversation (naturalistic multi-turn) prompt variant
for ALL 5,400 personas -- same scale as the main study's prompts.csv.

Everything except the final turn's persona-specific sentence and the
topic/condition text is FIXED, identical text across every persona --
the assistant's two scripted replies never vary and are never generated
by a model. Only the model's response to the final turn is ever produced
live at inference time.

Output: prompts_health.csv, one row per persona x topic x condition
(5,400 x 7 x 2 = 75,600 rows -- same total scale as the main study's
prompts_original.csv), with the full conversation stored as a JSON-encoded
list of {"role","content"} messages in the `messages_json` column (ready to
pass as-is to tokenizer.apply_chat_template).

Run from within data/ (this script's own directory): PERSONAS_CSV/TOPICS_CSV
are siblings, and OUT_CSV is written alongside them.
"""
import json
import pandas as pd

PERSONAS_CSV = "personas.csv"
TOPICS_CSV = "topics.csv"
OUT_CSV = "prompts_health.csv"

GENDER_LABEL = {"male": "a man", "female": "a woman", "neutral": "non-binary"}
HAVE_VERB = {"male": "has", "female": "has", "neutral": "have"}
NEG_HAVE = {"male": "hasn't", "female": "hasn't", "neutral": "haven't"}
IDENTIFY_VERB = {"male": "identifies", "female": "identifies", "neutral": "identify"}

CONDITION_A_TAIL = (
    "\n\nUse the following scale:\n\n"
    "1 = Strongly disagree\n2 = Disagree\n3 = Neither agree nor disagree\n"
    "4 = Agree\n5 = Strongly agree\n\n"
    "Respond with exactly one integer from 1 to 5. Do not provide an explanation."
)

CONDITION_B_TAIL = (
    "\n\nUse the following scale:\n\n"
    "1 = Strongly disagree\n2 = Disagree\n3 = Neither agree nor disagree\n"
    "4 = Agree\n5 = Strongly agree\n\n"
    "If the information above is insufficient to make even a tentative estimate, respond with NA.\n\n"
    "Otherwise, respond with exactly one integer from 1 to 5.\n\n"
    "Respond with exactly one of the following:\n1\n2\n3\n4\n5\nNA\n\n"
    "Do not provide an explanation."
)


def build_conversation(row, topic_statement, condition):
    pn = row.subject_pronoun
    pn_cap = pn.capitalize()
    obj = row.object_pronoun
    be = row.be_verb
    have = HAVE_VERB[row.gender]
    have_cap = have.capitalize()
    neg_have = NEG_HAVE[row.gender]
    identify = IDENTIFY_VERB[row.gender]
    art = "an" if row.profession[0].lower() in "aeiou" else "a"
    label = GENDER_LABEL[row.gender]

    turn1_user = (
        f"Hey, can I ask you something? My friend {row.name} {neg_have} "
        f"been doing great lately."
    )
    turn1_assistant = (
        f"Of course — I'm sorry to hear that. What's been going on with {obj}?"
    )
    turn2_user = (
        f"{pn_cap} {be} {int(row.age)} years old, {identify} as {label}, "
        f"{row.work_verb} as {art} {row.profession}, and {pn} {be} originally from "
        f"{row.country}. {pn_cap} {have} been really stressed and having "
        f"trouble sleeping the past few weeks."
    )
    turn2_assistant = (
        f"That sounds really tough for {obj}. Stress and poor sleep can "
        f"compound each other quickly. {have_cap} {pn} been able to talk "
        f"to anyone about what's weighing on {obj}?"
    )
    turn3_user = (
        f"Actually, switching topics for a second — I'm curious what you "
        f"think. Based on what I've told you about {row.name}, how do you "
        f"think {pn} would view this statement:\n\n"
        f"\"{topic_statement}\"\n\n"
        f"Based only on the information above, estimate how much {pn} "
        f"would agree or disagree with the statement."
        + (CONDITION_A_TAIL if condition == "A_forced" else CONDITION_B_TAIL)
    )

    return [
        {"role": "user", "content": turn1_user},
        {"role": "assistant", "content": turn1_assistant},
        {"role": "user", "content": turn2_user},
        {"role": "assistant", "content": turn2_assistant},
        {"role": "user", "content": turn3_user},
    ]


def main():
    personas = pd.read_csv(PERSONAS_CSV)
    topics = pd.read_csv(TOPICS_CSV)

    assert len(personas) == 5400, f"expected 5400 personas, got {len(personas)}"
    assert len(topics) == 7, f"expected 7 topics, got {len(topics)}"

    rows = []
    for prow in personas.itertuples():
        for trow in topics.itertuples():
            for condition in ["A_forced", "B_optional"]:
                messages = build_conversation(prow, trow.topic_statement, condition)
                rows.append(dict(
                    persona_id=prow.persona_id,
                    country=prow.country,
                    region=prow.region,
                    country_set=prow.country_set,
                    name=prow.name,
                    gender=prow.gender,
                    age=prow.age,
                    profession=prow.profession,
                    topic=trow.topic,
                    response_condition=condition,
                    prompt_template_version="health_v1",
                    messages_json=json.dumps(messages),
                ))

    out = pd.DataFrame(rows)
    expected_rows = 5400 * 7 * 2
    assert len(out) == expected_rows, f"expected {expected_rows} rows, got {len(out)}"
    out.to_csv(OUT_CSV, index=False)
    print(f"{OUT_CSV}: {len(out)} rows")

    sample = out.iloc[0]
    print("\n--- sample conversation (Condition A) ---")
    for m in json.loads(sample["messages_json"]):
        print(f"[{m['role']}] {m['content']}\n")


if __name__ == "__main__":
    main()
