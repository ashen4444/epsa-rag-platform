"""Fixed Component 01 examples retained while later EPSA components are added."""

import pytest

from epsa_rag.epsa.question_analysis import AnswerType, QuestionType, RuleBasedQuestionAnalyzer


@pytest.mark.parametrize(
    ("question", "targets"),
    [
        ("Which mountain is higher, Tongshanjiabu or Himalchuli?", ["Tongshanjiabu", "Himalchuli"]),
        ("Who is older, Hampton Del Ruth or Ted Kotcheff?", ["Hampton Del Ruth", "Ted Kotcheff"]),
        (
            "Which skyscraper has more floors, SNCI Tower or 8 Spruce Street?",
            ["SNCI Tower", "8 Spruce Street"],
        ),
        (
            "Which was released first, Gasland or To Shoot an Elephant?",
            ["Gasland", "To Shoot an Elephant"],
        ),
        (
            "Was Hoobastank or Fountains of Wayne formed first?",
            ["Hoobastank", "Fountains of Wayne"],
        ),
        (
            "Does Bazhong or Kaiyuan, Liaoning have a higher population?",
            ["Bazhong", "Kaiyuan, Liaoning"],
        ),
        ("Which of these artists is older, Warrel Dane or Roy Khan?", ["Warrel Dane", "Roy Khan"]),
        (
            "Between two bands Audioslave and Bodyjar, which group formed earlier?",
            ["Audioslave", "Bodyjar"],
        ),
    ],
)
def test_scalar_comparisons_take_precedence_and_extract_two_clean_targets(
    question: str, targets: list[str]
) -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(question)

    assert analysis.question_type is QuestionType.COMPARISON
    assert [target.text for target in analysis.comparison_targets] == targets


@pytest.mark.parametrize(
    "question",
    [
        "Are Daryl Hall and Gerry Marsden both musicians?",
        "Are Pterostyrax and Dregea both native to Asia?",
        "Was either Craig Melville or Grover Jones born in Indiana?",
    ],
)
def test_genuine_boolean_propositions_are_not_comparisons(question: str) -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(question)

    assert analysis.question_type is QuestionType.YES_NO
    assert analysis.expected_answer_type is AnswerType.BOOLEAN


def test_temporal_between_interval_is_not_a_comparison() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(
        "Which member conducted research between 1997 and 2005?"
    )

    assert analysis.question_type is not QuestionType.COMPARISON
    assert analysis.comparison_targets == ()


@pytest.mark.parametrize(
    ("question", "answer_type"),
    [
        ("Operation Cold Comfort was founded in what year?", AnswerType.DATE),
        ("What is the birthdate of this Uruguayan former footballer?", AnswerType.DATE),
        ("When did the director of Madadayo die?", AnswerType.DATE),
        ("The university is located in what city?", AnswerType.LOCATION),
        ("In what county is the General Motors Technical Center located?", AnswerType.LOCATION),
        (
            "Which town with a population of 5,457 is located by Mount Monadnock?",
            AnswerType.LOCATION,
        ),
    ],
)
def test_explicit_answer_slots_override_descriptive_body_keywords(
    question: str, answer_type: AnswerType
) -> None:
    assert RuleBasedQuestionAnalyzer().analyze(question).expected_answer_type is answer_type


@pytest.mark.parametrize(
    "question",
    [
        'What year did the director of "Madadayo" die?',
        "When was the actor born who starred in Sugarfoot?",
        "When did the coach of the 1963 Oklahoma Sooners football team die?",
        (
            "Where is the stadium at which the 1964 Georgia Tech Yellow Jackets football team "
            "played located?"
        ),
        "Operation Cold Comfort was a raid by a special forces unit founded in what year?",
    ],
)
def test_compositional_structures_are_bridges(question: str) -> None:
    assert RuleBasedQuestionAnalyzer().analyze(question).question_type is QuestionType.BRIDGE


def test_entities_are_unicode_safe_and_do_not_fragment_multiword_names() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(
        "Did Yasujirō Ozu or Alfonso Cuarón direct Götterdämmerung?"
    )

    assert {entity.text for entity in analysis.seed_entities} == {
        "Yasujirō Ozu",
        "Alfonso Cuarón",
        "Götterdämmerung",
    }


def test_titles_and_plain_prepositions_do_not_create_false_relation_hints() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(
        'Was "How I Met Your Mother" about the father of the atomic bomb in a film?'
    )

    assert "parent" not in {hint.relation for hint in analysis.required_relation_hints}
    assert "located" not in {hint.relation for hint in analysis.required_relation_hints}


def test_strong_location_relation_and_relation_deduplication_are_preserved() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(
        "Where was Ada born and where is her home city located in France?"
    )

    assert [hint.relation for hint in analysis.required_relation_hints] == ["born", "located"]


def test_documented_bridge_example_retains_its_structured_contract() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze("Where was the director of Inception born?")

    assert analysis.question_type is QuestionType.BRIDGE
    assert analysis.expected_answer_type is AnswerType.LOCATION
    assert "inception" in {entity.normalized for entity in analysis.seed_entities}
    assert {hint.relation for hint in analysis.required_relation_hints} == {"born", "directed"}
    assert analysis.comparison_targets == ()
    assert analysis.answer_type_candidates[0].confidence == 0.75


@pytest.mark.parametrize(
    ("question", "answer_type"),
    [
        ("Who designed the theater where LPO plays?", AnswerType.PERSON),
        ("How many episodes are in the series where the actor appeared?", AnswerType.NUMBER),
        ("Which university where Ada studied has how many employees?", AnswerType.NUMBER),
        ("What is the population of the city where the band is based?", AnswerType.NUMBER),
        (
            "What is the founding year of the company in the Netherlands where it began?",
            AnswerType.DATE,
        ),
    ],
)
def test_answer_slots_take_precedence_over_descriptive_clause_words(
    question: str, answer_type: AnswerType
) -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(question)

    assert analysis.expected_answer_type is answer_type


def test_actress_relative_clause_is_a_date_bridge() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(
        'In what year was actress who starred in "Streak" with Rumer Willis born?'
    )

    assert analysis.question_type is QuestionType.BRIDGE
    assert analysis.expected_answer_type is AnswerType.DATE


@pytest.mark.parametrize(
    ("question", "targets"),
    [
        (
            "Which case was brought to court first Miller v. California or Gates v. Collier?",
            ["Miller v. California", "Gates v. Collier"],
        ),
        (
            "Which casino closed first The Atlantic Club Casino Hotel or Trump Plaza Hotel "
            "and Casino?",
            ["The Atlantic Club Casino Hotel", "Trump Plaza Hotel and Casino"],
        ),
        ("Which genus has more species, Bactris and Epigaea?", ["Bactris", "Epigaea"]),
    ],
)
def test_scalar_comparisons_return_clean_targets_and_never_boolean(
    question: str, targets: list[str]
) -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(question)

    assert analysis.question_type is QuestionType.COMPARISON
    assert analysis.expected_answer_type is not AnswerType.BOOLEAN
    assert [target.text for target in analysis.comparison_targets] == targets


@pytest.mark.parametrize(
    "question",
    [
        "What officially ended the first phase of the conflict between British Raj and Emirate "
        "in 1878?",
        "David Wayne Hull, born 1962 or 1963, was the first Grand Wizard of what organization?",
    ],
)
def test_ordinal_and_alternative_prose_do_not_imply_scalar_comparison(question: str) -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(question)
    assert analysis.question_type is not QuestionType.COMPARISON


def test_leading_auxiliary_is_not_retained_as_a_seed_entity() -> None:
    seeds = RuleBasedQuestionAnalyzer().analyze(
        "Were Halldór Laxness and Sigríður Undset born?"
    ).seed_entities

    assert "Were Halldór Laxness" not in {seed.text for seed in seeds}
    assert "Halldór Laxness" in {seed.text for seed in seeds}


def test_metaphorical_parent_words_are_not_relation_hints_but_named_kinship_is() -> None:
    metaphor = RuleBasedQuestionAnalyzer().analyze(
        "Who was called the father of modern organized crime?"
    )
    kinship = RuleBasedQuestionAnalyzer().analyze("Who was the father of Edward Garrison Walker?")

    assert "parent" not in {hint.relation for hint in metaphor.required_relation_hints}
    assert "parent" in {hint.relation for hint in kinship.required_relation_hints}


@pytest.mark.parametrize(
    ("question", "targets"),
    [
        (
            "What took place first, The Korean War or The Western Allied invasion of Germany?",
            ["The Korean War", "The Western Allied invasion of Germany"],
        ),
        (
            "Which one between Milium and Eucommia lives more widely spread across the world?",
            ["Milium", "Eucommia"],
        ),
        (
            "Which was founded first, University of California, Santa Barbara or Hamdard "
            "University?",
            ["University of California, Santa Barbara", "Hamdard University"],
        ),
        (
            "Who was born more recently, Flula Borg, or Dirk Nowitzki?",
            ["Flula Borg", "Dirk Nowitzki"],
        ),
        (
            "Is the Metropolitan Tower or 750 7th Avenue Skyscraper taller?",
            ["the Metropolitan Tower", "750 7th Avenue Skyscraper"],
        ),
    ],
)
def test_final_comparison_target_boundaries_are_clean(question: str, targets: list[str]) -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(question)

    assert analysis.question_type is QuestionType.COMPARISON
    assert [target.text for target in analysis.comparison_targets] == targets


def test_relative_words_scattered_through_prose_do_not_make_a_comparison() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(
        "Who was the king of England who became king after the death of his younger brother "
        "and whose reign was chaotic due to rivalry with relatives until his own death a few "
        "decades later?"
    )

    assert analysis.question_type is not QuestionType.COMPARISON
    assert analysis.comparison_targets == ()


def test_requested_population_overrides_generic_entity_but_population_descriptor_does_not() -> None:
    requested = RuleBasedQuestionAnalyzer().analyze(
        "What is the 2010 census population of the county where Wildcat Brook flows through "
        "Jackson?"
    )
    descriptor = RuleBasedQuestionAnalyzer().analyze(
        "Which town in New Hampshire with a population of 5,457 in 2010 is located by Mount "
        "Monadnock?"
    )

    assert requested.expected_answer_type is AnswerType.NUMBER
    assert descriptor.expected_answer_type is AnswerType.LOCATION


def test_typed_noun_phrase_uses_its_semantic_head_for_a_comparison_answer() -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(
        "Which film director is younger, Manoel de Oliveira or Alfonso Cuarón?"
    )

    assert analysis.question_type is QuestionType.COMPARISON
    assert analysis.expected_answer_type is AnswerType.PERSON


def test_possessive_title_seed_is_preserved_without_a_leading_s_fragment() -> None:
    seeds = RuleBasedQuestionAnalyzer().analyze("How did Pre's Trail's namesake die?").seed_entities

    assert "s Trail" not in {seed.text for seed in seeds}
    assert "Pre's Trail" in {seed.text for seed in seeds}


@pytest.mark.parametrize("question", ["Love Child is a 1982 biopic.", "Father Ted is a comedy."])
def test_title_like_spans_do_not_emit_parent_or_child_relation_hints(question: str) -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(question)
    relations = {hint.relation for hint in analysis.required_relation_hints}

    assert "parent" not in relations
    assert "child" not in relations
