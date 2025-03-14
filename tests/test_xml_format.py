import pytest

from pydantic_xml import BaseXmlModel, element

from pydantic import ValidationError


class ArticleResponse1XML(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question."""

    title: str = element(description="Title of the article")
    answer: str = element(
        description="Provide a detailed description of historical events to answer the question"
    )
    number: int = element(description="A number that is most relevant to the question.")


# Lists of simple types
class ArticleResponse2XML(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question."""

    title: str = element(description="Title of the article")
    answer: str = element(description="Answer the writer's question")
    further_questions: list[str] = element(
        tag="further_question",
        description="A list of related questions that may be of interest to the readers.",
    )


# Nested types
class HistoricalEventXML(BaseXmlModel):
    """The year and explanation of a historical event."""

    year: int = element(description="The year of the historical event")
    event: str = element(
        description="A clear and concise explanation of what happened in this event"
    )


class ArticleResponse3XML(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question."""

    title: str = element(description="[Title of the article]")
    historical_event_1: HistoricalEventXML = element(
        description="A first historical event relevant to the question"
    )
    historical_event_2: HistoricalEventXML = element(
        description="A second historical event relevant to the question"
    )


# Lists of custom types
class ArticleResponse4XML(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question."""

    title: str = element(description="Title of the article")
    historical_timeline: list[HistoricalEventXML] = element(
        description="A list of historical events relevant to the question"
    )


def test_valid_xml():
    valid_xml = """
<article>
  <title>[Title of the article]</title>
  <historical_event_1>
    <year>1962</year>
    <event>[A clear description of what happened in this event]</event>
  </historical_event_1>
  <historical_event_2>
    <year>1911</year>
    <event>[A clear description of what happened in this event]</event>
  </historical_event_2>
</article>
"""
    assert ArticleResponse3XML.from_xml(valid_xml) is not None


def test_invalid_title_tag():
    invalid_xml = """
<article>
  <problem>[Title of the article]</problem>
  <historical_event_1>
    <year>1962</year>
    <event>[A clear description of what happened in this event]</event>
  </historical_event_1>
  <historical_event_2>
    <year>1911</year>
    <event>[A clear description of what happened in this event]</event>
  </historical_event_2>
</article>
"""
    with pytest.raises(ValidationError):
        ArticleResponse3XML.from_xml(invalid_xml)


def test_malformed_xml():
    malformed_xml = """
<article>
  <title>[Title of the article]</title>
  <historical_event_1>
    <year>1962 AD
    <event>[A clear description of what happened in this event]</event>
  </historical_event_1>
  <historical_event_2>
    <year>1911</year>
    <event>[A clear description of what happened in this event]</event>
  </historical_event_2>
</article>
"""
    with pytest.raises(Exception, match="Opening and ending tag mismatch"):
        ArticleResponse3XML.from_xml(malformed_xml)


def test_mismatched_tags():
    mismatched_xml = """
<article>
  <title>[Title of the article]</title>
  <historical_event_1>
    <year>1962 AD</year>
    <event>[A clear description of what happened in this event]</event>
  </historical_event_1>
  <historical_event_2>
    <year>1911</year>
    <event>[A clear description of what happened in this event]</event>
  </historical_event_3>
</article>
"""
    with pytest.raises(Exception, match="Opening and ending tag mismatch"):
        ArticleResponse3XML.from_xml(mismatched_xml)


def valid_xml_2():
    test_xml = """
<article>
  <title>
    Title of the article
  </title>
  <answer>
    Answer the writer's question
  </answer>
  <further_question>
    A list of related questions that may be of interest to the readers.
  </further_question>
  <further_question>
    A list of related questions that may be of interest to the readers.
  </further_question>
</article>
"""
    assert ArticleResponse2XML.from_xml(test_xml) is not None


def test_incorrect_schema_2():
    test_xml = """
<article>
  <title>
    Title of the article
  </title>
  <query>
    Answer the writer's question
  </query>
  <further_question>
    A list of related questions that may be of interest to the readers.
  </further_question>
  <further_question>
    A list of related questions that may be of interest to the readers.
  </further_question>
</article>
"""
    with pytest.raises(ValidationError):
        ArticleResponse2XML.from_xml(test_xml)


def test_mismatched_tags_2():
    test_xml = """
<article>
  <title>
    Title of the article
  </title>
  <further_question>
    A list of related questions that may be of interest to the readers.
  </further_question>
  <further_question>
    A list of related questions that may be of interest to the readers.
  </answer>
</article>
"""
    with pytest.raises(Exception, match="Opening and ending tag mismatch"):
        ArticleResponse2XML.from_xml(test_xml)


def test_valid_xml_4():
    test_xml = """
<article>
  <title>
    Title of the article
  </title>
<historical_timeline>
  <year>
    1943
  </year>
  <event>
    A clear and concise explanation of what happened in this event
  </event>
</historical_timeline>
<historical_timeline>
  <year>
    1943
  </year>
  <event>
    A clear and concise explanation of what happened in this event
  </event>
</historical_timeline>
</article>
"""
    assert ArticleResponse4XML.from_xml(test_xml) is not None


def test_comments_4():
    test_xml = """
<article>
  <!--Structured article for publication answering a reader's question.-->
  <title>
    {Title of the article}
  </title>
<!-- First list element -->
<historical_timeline>
  <!--The year and explanation of a historical event.-->
  <year>
    1943
  </year>
  <event>
    {A clear and concise explanation of what happened in this event}
  </event>
</historical_timeline>
<!-- Next list element -->
<historical_timeline>
  <!--The year and explanation of a historical event.-->
  <year>
    1943
  </year>
  <event>
    {A clear and concise explanation of what happened in this event}
  </event>
</historical_timeline>
</article>
"""
    assert ArticleResponse4XML.from_xml(test_xml) is not None


def test_comments_4():
    test_xml = """
<article>
  <!--Structured article for publication answering a reader's question.-->
  <title>
    {Title of the article}
  </title>
<!-- First list element -->
<historical_timeline>
  <!--The year and explanation of a historical event.-->
  <year>
    1943
  <event>
  </year>
    {A clear and concise explanation of what happened in this event}
  </event>
</historical_timeline>
<!-- Next list element -->
<historical_timeline>
  <!--The year and explanation of a historical event.-->
  <year>
    1943
  </year>
  <event>
    {A clear and concise explanation of what happened in this event}
  </event>
</historical_timeline>
</article>
"""
    with pytest.raises(Exception, match="Opening and ending tag mismatch"):
        ArticleResponse2XML.from_xml(test_xml)
