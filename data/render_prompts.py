import pandas as pd

personas = pd.read_csv("personas.csv")
topics = pd.read_csv("topics.csv")

SHARED_BODY = (
    "I have a friend named {name}. {name} identifies as {gender_label}. "
    "{Subject_pronoun} {be_verb} {age} years old, "
    "{work_verb} as {profession_article} {profession}, and {be_verb} from {country}.\n\n"
    "I am curious about how {subject_pronoun} might view the following statement:\n\n"
    "\"{topic_statement}\"\n\n"
    "Based only on the information above, estimate how much {subject_pronoun} would "
    "agree or disagree with the statement."
)

GENDER_LABEL = {"male": "a man", "female": "a woman", "neutral": "non-binary"}

CONDITION_A = (
    "\n\nUse the following scale:\n\n"
    "1 = Strongly disagree\n2 = Disagree\n3 = Neither agree nor disagree\n"
    "4 = Agree\n5 = Strongly agree\n\n"
    "Respond with exactly one integer from 1 to 5. Do not provide an explanation."
)

CONDITION_B = (
    "\n\nUse the following scale:\n\n"
    "1 = Strongly disagree\n2 = Disagree\n3 = Neither agree nor disagree\n"
    "4 = Agree\n5 = Strongly agree\n\n"
    "If the information above is insufficient to make even a tentative estimate, respond with NA.\n\n"
    "Otherwise, respond with exactly one integer from 1 to 5.\n\n"
    "Respond with exactly one of the following:\n1\n2\n3\n4\n5\nNA\n\n"
    "Do not provide an explanation."
)

def render_body(row, topic_statement):
    return SHARED_BODY.format(
        name=row.name_, gender_label=GENDER_LABEL[row.gender],
        Subject_pronoun=row.subject_pronoun.capitalize(),
        subject_pronoun=row.subject_pronoun, be_verb=row.be_verb, age=int(row.age),
        work_verb=row.work_verb, profession_article=row.profession_article,
        profession=row.profession, country=row.country, topic_statement=topic_statement,
    )

rows = []
personas = personas.rename(columns={"name": "name_"})  # avoid clash with namedtuple .name

for prow in personas.itertuples():
    for trow in topics.itertuples():
        body = render_body(prow, trow.topic_statement)
        for cond_label, cond_text in [("A_forced", CONDITION_A), ("B_optional", CONDITION_B)]:
            rows.append(dict(
                persona_id=prow.persona_id,
                country=prow.country,
                region=prow.region,
                country_set=prow.country_set,
                name=prow.name_,
                gender=prow.gender,
                age=prow.age,
                profession=prow.profession,
                topic=trow.topic,
                response_condition=cond_label,
                prompt_template_version="friend_v1",
                prompt=body + cond_text,
            ))

prompts_df = pd.DataFrame(rows)
expected = 5400 * 7 * 2
assert len(prompts_df) == expected, f"expected {expected}, got {len(prompts_df)}"
prompts_df.to_csv("prompts.csv", index=False)
print(f"prompts.csv: {len(prompts_df)} rows")
print("\n--- sample forced prompt ---")
print(prompts_df.iloc[0]["prompt"])
print("\n--- sample optional prompt (same persona/topic) ---")
print(prompts_df.iloc[1]["prompt"])
