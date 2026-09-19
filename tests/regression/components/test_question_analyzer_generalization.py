"""Comparison structures must generalize across wording without absorbing factoids."""

from __future__ import annotations

import pytest

from epsa_rag.epsa.question_analysis import AnswerType, QuestionType, RuleBasedQuestionAnalyzer


@pytest.mark.parametrize(
    ("question", "targets"),
    [
        ("Who won more awards, Frank Sinatra or Nivek Ogre?", ("Frank Sinatra", "Nivek Ogre")),
        (
            "Who has released more solo albums, Nick Carter or Brady Seals?",
            ("Nick Carter", "Brady Seals"),
        ),
        (
            "Who has written more #1 songs, Chris DeStefano or Brett Eldredge?",
            ("Chris DeStefano", "Brett Eldredge"),
        ),
        ("Who has more victories Chelsea or Manchester United?", ("Chelsea", "Manchester United")),
        (
            "How has played in more bands, Kim Wilson or Chino Moreno?",
            ("Kim Wilson", "Chino Moreno"),
        ),
        ("Does Erodium or Cymbidium include more species?", ("Erodium", "Cymbidium")),
        (
            "Which long-established US university is older: University of California, "
            "Berkeley or Syracuse University?",
            ("University of California, Berkeley", "Syracuse University"),
        ),
        (
            "Who is older out of Bob Saget, the American comedian, and Indian director S. Shankar?",
            ("Bob Saget", "S. Shankar"),
        ),
        (
            "Who was born first out of Leopold Lummerstorfer and Laurent Touil-Tartour?",
            ("Leopold Lummerstorfer", "Laurent Touil-Tartour"),
        ),
        (
            "Which battle took place closer to New York City, Battle of Fredericksburg or "
            "the Western Allied invasion of Germany?",
            ("Battle of Fredericksburg", "the Western Allied invasion of Germany"),
        ),
        (
            "Which European Space Agency astronaut flew more Space Shuttle missions, Hans "
            "Schlegel of Germany or first Belgian in space Dirk Frimout?",
            ("Hans Schlegel", "Dirk Frimout"),
        ),
        (
            "Which band has released the most studio albums, Blonde Redhead or Rob Zombie's "
            "band White Zombie?",
            ("Blonde Redhead", "White Zombie"),
        ),
        (
            "Which avalanche happened first, Rigopiano avalanche or 1999 Galtür avalanche?",
            ("Rigopiano avalanche", "1999 Galtür avalanche"),
        ),
        ("Who had more members, X or The La's?", ("X", "The La's")),
        (
            "The publication of which magazine ended first, Right On! or Castle of Frankenstein?",
            ("Right On!", "Castle of Frankenstein"),
        ),
        (
            "Which movie did Disney produce first, The Many Adventures of Winnie the Pooh or "
            "Ride a Wild Pony?",
            ("The Many Adventures of Winnie the Pooh", "Ride a Wild Pony"),
        ),
        (
            "Which film was released first out of The Hunchback of Notre Dame and Miracle of "
            "the White Stallions?",
            ("The Hunchback of Notre Dame", "Miracle of the White Stallions"),
        ),
        (
            "Which writer, Karen Blixen or Woody Allen, has a broader range of artistic talents?",
            ("Karen Blixen", "Woody Allen"),
        ),
    ],
)
def test_comparison_families_return_semantic_targets(
    question: str, targets: tuple[str, str]
) -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(question)

    assert analysis.question_type is QuestionType.COMPARISON
    assert analysis.expected_answer_type is not AnswerType.BOOLEAN
    assert tuple(target.text for target in analysis.comparison_targets) == targets
    assert len({target.normalized for target in analysis.comparison_targets}) == 2
    assert all(
        analysis.normalized_question[target.start : target.end] == target.text.casefold()
        for target in analysis.comparison_targets
    )


@pytest.mark.parametrize(
    "question",
    [
        "Who won more awards, Ada Lovelace or Grace Hopper?",
        "Who has more awards, Ada Lovelace or Grace Hopper?",
        "Which person has more awards, Ada Lovelace or Grace Hopper?",
        "Between Ada Lovelace and Grace Hopper, who has more awards?",
        "Which was released first, Gasland or To Shoot an Elephant?",
        "Which came out first, Gasland or To Shoot an Elephant?",
        "Which was released earlier, Gasland or To Shoot an Elephant?",
        "Between Gasland and To Shoot an Elephant, which was released first?",
        "Who was born more recently, Ada Lovelace or Grace Hopper?",
        "Between Ada Lovelace and Grace Hopper, who was born more recently?",
        "Which singer was born more recently, Ada Lovelace or Grace Hopper?",
    ],
)
def test_comparison_paraphrases_keep_the_same_two_targets(question: str) -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(question)

    assert analysis.question_type is QuestionType.COMPARISON
    expected = (
        ("Ada Lovelace", "Grace Hopper")
        if "Ada Lovelace" in question
        else ("Gasland", "To Shoot an Elephant")
    )
    assert tuple(target.text for target in analysis.comparison_targets) == expected


@pytest.mark.parametrize(
    ("question", "expected_type"),
    [
        ('What type of film are both "500 Years Later" and "Manson"?', QuestionType.FACTOID),
        ("What type of film are Higher Ground and Manson?", QuestionType.FACTOID),
        ("What profession do Alex Beard and Tony Hall have in common?", QuestionType.FACTOID),
        ("Are Daryl Hall and Gerry Marsden both musicians?", QuestionType.YES_NO),
        ("Do Bloody Mary and Sidecar share any ingredients?", QuestionType.YES_NO),
        (
            "Which goalkeeper was nicknamed the Black Spider, Turgay Şeren or Lev Yashin?",
            QuestionType.FACTOID,
        ),
        ("Which dog originated in Galicia, A or B?", QuestionType.FACTOID),
        (
            "What kind of building did the Romans use that is still the most common "
            "architectural style for churches in Europe and America?",
            QuestionType.FACTOID,
        ),
        (
            "Who was the first Russian composer to make a lasting impression internationally, "
            "Alessandro Scarlatti or Pyotr Ilyich Tchaikovsky?",
            QuestionType.FACTOID,
        ),
        (
            "What American indie rock group first had Jesse Sandoval as the drummer and now "
            "has Jon Sortland on drums?",
            QuestionType.FACTOID,
        ),
        (
            "Which member of the research project conducted by the Creation Research Society "
            "and the Institute for Creation Research between 1997 and 2005 was Kevin R. Henke "
            "most critical of?",
            QuestionType.FACTOID,
        ),
    ],
)
def test_shared_property_and_candidate_selection_are_not_scalar_comparisons(
    question: str, expected_type: QuestionType
) -> None:
    analysis = RuleBasedQuestionAnalyzer().analyze(question)

    assert analysis.question_type is expected_type
    assert analysis.comparison_targets == ()
    if expected_type is QuestionType.YES_NO:
        assert analysis.expected_answer_type is AnswerType.BOOLEAN


@pytest.mark.parametrize(
    ("question", "expected_type"),
    [
        (
            "According to the 2010 census, what was the population of the city in which "
            "Boyle's Thirty Acres was located?",
            AnswerType.NUMBER,
        ),
        (
            "Which town in New Hampshire with a population of 5,457 in 2010 is located by "
            "Mount Monadnock?",
            AnswerType.LOCATION,
        ),
        ("The city's population was what in 2010?", AnswerType.NUMBER),
        ("What population did the city report in 2010?", AnswerType.NUMBER),
        (
            "Which American film director is an advisor for Disney, Rick Ray or John Lasseter?",
            AnswerType.PERSON,
        ),
        ("Which South Korean actress is older, Kim A or Lee B?", AnswerType.PERSON),
        ("Which British singer-songwriter is older, Kim A or Lee B?", AnswerType.PERSON),
        (
            "Dick Miller acted in which movie alongside actor Arnold Schwarzenegger?",
            AnswerType.TITLE_OR_WORK,
        ),
    ],
)
def test_explicit_answer_slots_with_modifiers(
    question: str, expected_type: AnswerType
) -> None:
    assert RuleBasedQuestionAnalyzer().analyze(question).expected_answer_type is expected_type
