import pandas as pd
import itertools

# ============================================================
# 1. NAMES.CSV
# ============================================================
names_rows = [
    # ORIGINAL 10 COUNTRIES — male/female pulled directly from existing project file
    dict(country="Germany", region="Europe", country_set="original_10", gender="male", name="Thomas", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Germany", region="Europe", country_set="original_10", gender="female", name="Emma", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Germany", region="Europe", country_set="original_10", gender="neutral", name="Robin", validation_status="strong", evidence_type="registry/aggregator", usage_skew="roughly balanced", notes="established German unisex name"),

    dict(country="France", region="Europe", country_set="original_10", gender="male", name="Pierre", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="France", region="Europe", country_set="original_10", gender="female", name="Camille", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="France", region="Europe", country_set="original_10", gender="neutral", name="Dominique", validation_status="strong", evidence_type="registry/aggregator", usage_skew="roughly balanced", notes="classic French unisex name; Camille was already the female name in this dataset"),

    dict(country="USA", region="North America", country_set="original_10", gender="male", name="Tyler", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="USA", region="North America", country_set="original_10", gender="female", name="Ashley", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="USA", region="North America", country_set="original_10", gender="neutral", name="Taylor", validation_status="strong", evidence_type="registry (SSA)", usage_skew="roughly balanced", notes="well-documented US unisex name"),

    dict(country="Brazil", region="Latin America", country_set="original_10", gender="male", name="Carlos", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Brazil", region="Latin America", country_set="original_10", gender="female", name="Ana", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Brazil", region="Latin America", country_set="original_10", gender="neutral", name="Ariel", validation_status="acceptable", evidence_type="cultural/literature", usage_skew="unverified", notes="used across genders in Brazil"),

    dict(country="Kenya", region="Africa", country_set="original_10", gender="male", name="James", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Kenya", region="Africa", country_set="original_10", gender="female", name="Aisha", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Kenya", region="Africa", country_set="original_10", gender="neutral", name="Amani", validation_status="acceptable", evidence_type="cultural/literature", usage_skew="unverified", notes="Swahili 'peace', used across genders"),

    dict(country="Egypt", region="MENA", country_set="original_10", gender="male", name="Omar", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Egypt", region="MENA", country_set="original_10", gender="female", name="Fatima", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Egypt", region="MENA", country_set="original_10", gender="neutral", name="Nour", validation_status="strong", evidence_type="cultural/literature", usage_skew="balanced", notes="transliteration variant of Pakistan's 'Noor' -- intentional, documented"),

    dict(country="India", region="South Asia", country_set="original_10", gender="male", name="Raj", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="India", region="South Asia", country_set="original_10", gender="female", name="Priya", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="India", region="South Asia", country_set="original_10", gender="neutral", name="Kiran", validation_status="strong", evidence_type="cultural/literature, widely documented", usage_skew="balanced", notes="textbook Indian unisex name"),

    dict(country="Japan", region="East Asia", country_set="original_10", gender="male", name="Kenji", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Japan", region="East Asia", country_set="original_10", gender="female", name="Yuki", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Japan", region="East Asia", country_set="original_10", gender="neutral", name="Hikaru", validation_status="acceptable", evidence_type="cultural/literature", usage_skew="unverified", notes="genuinely usable across genders"),

    dict(country="China", region="East Asia", country_set="original_10", gender="male", name="Wei", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="China", region="East Asia", country_set="original_10", gender="female", name="Mei", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="China", region="East Asia", country_set="original_10", gender="neutral", name="Yu", validation_status="acceptable", evidence_type="cultural/literature", usage_skew="unverified", notes="'Wei' was already the male name in this dataset; Yu used across genders"),

    dict(country="Mexico", region="Latin America", country_set="original_10", gender="male", name="Diego", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Mexico", region="Latin America", country_set="original_10", gender="female", name="Valentina", validation_status="strong", evidence_type="prior project data", usage_skew="n/a", notes="carried from v9 dataset"),
    dict(country="Mexico", region="Latin America", country_set="original_10", gender="neutral", name="Guadalupe", validation_status="strong", evidence_type="cultural/literature, widely documented", usage_skew="balanced", notes="famously borne across genders"),

    # NEW 10 COUNTRIES
    dict(country="Italy", region="Europe", country_set="added_10", gender="male", name="Marco", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Italy", region="Europe", country_set="added_10", gender="female", name="Giulia", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Italy", region="Europe", country_set="added_10", gender="neutral", name="Celeste", validation_status="acceptable", evidence_type="registry/aggregator", usage_skew="female-leaning", notes="attested cross-gender, likely skews female in practice"),

    dict(country="Poland", region="Europe", country_set="added_10", gender="male", name="Piotr", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Poland", region="Europe", country_set="added_10", gender="female", name="Katarzyna", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Poland", region="Europe", country_set="added_10", gender="neutral", name="Alex", validation_status="weak", evidence_type="literature", usage_skew="masculine-skewed internationally", notes="no traditional Polish neutral name exists; international form used pragmatically"),

    dict(country="Canada", region="North America", country_set="added_10", gender="male", name="Noah", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Canada", region="North America", country_set="added_10", gender="female", name="Olivia", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Canada", region="North America", country_set="added_10", gender="neutral", name="Riley", validation_status="acceptable", evidence_type="registry/aggregator", usage_skew="roughly balanced", notes="established unisex name in English-speaking Canada"),

    dict(country="Argentina", region="Latin America", country_set="added_10", gender="male", name="Mateo", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Argentina", region="Latin America", country_set="added_10", gender="female", name="Martina", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes="'Valentina' was already the female name for Mexico in this dataset"),
    dict(country="Argentina", region="Latin America", country_set="added_10", gender="neutral", name="Sol", validation_status="acceptable", evidence_type="cultural/literature", usage_skew="unverified, believed balanced", notes="replaces 'Cruz' (64/36 male-skewed per genderize.io aggregator data)"),

    dict(country="Nigeria", region="Africa", country_set="added_10", gender="male", name="Adebayo", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes="Yoruba"),
    dict(country="Nigeria", region="Africa", country_set="added_10", gender="female", name="Folake", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes="Yoruba"),
    dict(country="Nigeria", region="Africa", country_set="added_10", gender="neutral", name="Ayomide", validation_status="provisional", evidence_type="cultural/literature", usage_skew="unverified", notes="Yoruba, used across genders; distribution not yet documented"),

    dict(country="South Africa", region="Africa", country_set="added_10", gender="male", name="Sipho", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="South Africa", region="Africa", country_set="added_10", gender="female", name="Thandiwe", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="South Africa", region="Africa", country_set="added_10", gender="neutral", name="Thando", validation_status="provisional", evidence_type="cultural/literature", usage_skew="unverified", notes="replaces 'Themba', which skews male in practice"),

    dict(country="Morocco", region="MENA", country_set="added_10", gender="male", name="Youssef", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Morocco", region="MENA", country_set="added_10", gender="female", name="Khadija", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes="'Fatima' was already the female name for Egypt in this dataset"),
    dict(country="Morocco", region="MENA", country_set="added_10", gender="neutral", name="Ihsan", validation_status="provisional", evidence_type="cultural/literature", usage_skew="unverified", notes="kept pending further validation"),

    dict(country="Turkey", region="MENA", country_set="added_10", gender="male", name="Mehmet", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Turkey", region="MENA", country_set="added_10", gender="female", name="Elif", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Turkey", region="MENA", country_set="added_10", gender="neutral", name="Deniz", validation_status="strong", evidence_type="cultural/literature, widely documented", usage_skew="balanced", notes="textbook unisex Turkish name"),

    dict(country="Pakistan", region="South Asia", country_set="added_10", gender="male", name="Ahmed", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Pakistan", region="South Asia", country_set="added_10", gender="female", name="Ayesha", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="Pakistan", region="South Asia", country_set="added_10", gender="neutral", name="Noor", validation_status="strong", evidence_type="cultural/literature", usage_skew="balanced", notes="transliteration variant of Egypt's 'Nour' -- intentional, documented"),

    dict(country="South Korea", region="East Asia", country_set="added_10", gender="male", name="Min-jun", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="South Korea", region="East Asia", country_set="added_10", gender="female", name="Seo-yeon", validation_status="strong", evidence_type="registry/cultural convention", usage_skew="n/a", notes=""),
    dict(country="South Korea", region="East Asia", country_set="added_10", gender="neutral", name="Haneul", validation_status="acceptable", evidence_type="registry/aggregator", usage_skew="unverified, pure-Korean names typically more balanced than Sino-Korean", notes="replaces 'Ji-woo' (71/29 female-skewed per name-statistics aggregator)"),
]

names_df = pd.DataFrame(names_rows)
assert len(names_df) == 60, f"expected 60 name rows, got {len(names_df)}"
assert names_df.groupby("country")["gender"].nunique().eq(3).all()
# duplicate check within country
dupe_check = names_df.groupby("country")["name"].apply(lambda x: x.duplicated().any())
assert not dupe_check.any(), f"duplicate name within a country: {dupe_check[dupe_check].index.tolist()}"
names_df.to_csv("names.csv", index=False)
print(f"names.csv: {len(names_df)} rows, {names_df['country'].nunique()} countries")

# ============================================================
# 2. PROFESSIONS -- carried exactly from existing project file
# ============================================================
professions = ['accountant', 'administrative assistant', 'architect', 'bank teller',
    'business manager', 'cashier', 'chef', 'civil engineer', 'computer programmer',
    'construction worker', 'doctor', 'electrician', 'elementary school teacher',
    'farmer', 'flight attendant', 'hairdresser', 'janitor', 'journalist', 'lawyer',
    'mechanic', 'paramedic', 'pharmacist', 'plumber', 'police officer',
    'real estate agent', 'registered nurse', 'social worker', 'truck driver',
    'veterinarian', 'waiter']
assert len(professions) == 30

def article_for(word):
    return "an" if word[0].lower() in "aeiou" else "a"

# ============================================================
# 3. TOPICS -- draft statements, 3 carried, 4 new (DRAFT, needs your review)
# ============================================================
topics = [
    dict(topic="climate change", topic_statement="Climate change is a serious threat that requires immediate government action."),
    dict(topic="gender equality", topic_statement="Men and women should have equal rights and opportunities in all aspects of life."),
    dict(topic="immigration", topic_statement="Immigration is good for the country's economy and culture."),
    dict(topic="lgbtq rights", topic_statement="Homosexuality and same-sex relationships should be fully accepted by society."),
    dict(topic="religion and secularism", topic_statement="Religion should play an important role in guiding public policy."),
    dict(topic="economic redistribution", topic_statement="Government should redistribute wealth through heavy taxes on the rich."),
    dict(topic="trust in government", topic_statement="The government can generally be trusted to act in the best interest of the people."),
]
topics_df = pd.DataFrame(topics)
topics_df.to_csv("topics.csv", index=False)
print(f"topics.csv: {len(topics_df)} rows -- DRAFT WORDING, needs review before freezing")

ages = [25, 45, 65]
genders = ["male", "female", "neutral"]

# ============================================================
# 4. PERSONAS.CSV -- base grid, 20 countries x 3 genders x 3 ages x 30 professions
# ============================================================
names_lookup = {(r.country, r.gender): r.name for r in names_df.itertuples()}
region_lookup = {r.country: r.region for r in names_df.itertuples()}
countryset_lookup = {r.country: r.country_set for r in names_df.itertuples()}
countries = sorted(names_df["country"].unique())

pronoun_map = {
    "male":    dict(subject="he",   object="him", possessive="his",   be_verb="is",  work_verb="works"),
    "female":  dict(subject="she",  object="her", possessive="her",   be_verb="is",  work_verb="works"),
    "neutral": dict(subject="they", object="them", possessive="their", be_verb="are", work_verb="work"),
}

rows = []
pid = 0
for country in countries:
    for gender in genders:
        for age in ages:
            for profession in professions:
                pid += 1
                pr = pronoun_map[gender]
                rows.append(dict(
                    persona_id=f"P{pid:05d}",
                    country=country,
                    region=region_lookup[country],
                    country_set=countryset_lookup[country],
                    name=names_lookup[(country, gender)],
                    gender=gender,
                    age=age,
                    profession=profession,
                    profession_article=article_for(profession),
                    subject_pronoun=pr["subject"],
                    object_pronoun=pr["object"],
                    possessive_pronoun=pr["possessive"],
                    be_verb=pr["be_verb"],
                    work_verb=pr["work_verb"],
                ))

personas_df = pd.DataFrame(rows)
assert len(personas_df) == 20 * 3 * 3 * 30, f"expected 5400 rows, got {len(personas_df)}"
assert personas_df["persona_id"].is_unique
personas_df.to_csv("personas.csv", index=False)
print(f"personas.csv: {len(personas_df)} rows")

print("\nDone. Files: names.csv, topics.csv, personas.csv")
