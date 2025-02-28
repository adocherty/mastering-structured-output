from pydantic_xml import BaseXmlModel
from pydantic_xml import BaseXmlModel, element, attr

__all__ = [
    "ArticleResponse1XML",
    "ArticleResponse1nointXML",
    "ArticleResponse2XML",
    "ListofStrXML",
    "ArticleResponse2XMLalt",
    "HistoricalEventXML",
    "ArticleResponse3XML",
    "ArticleResponse4XML",
    "ListofHistoricalEventXML",
    "ArticleResponse4XMLalt",
]

## START: XML schema definitions


class ArticleResponse1XML(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="Title of the article")
    answer: str = element(
        description="Provide a detailed description of historical events to answer the question"
    )
    number: int = element(description="A number that is most relevant to the question.")


class ArticleResponse1nointXML(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="Title of the article")
    answer: str = element(
        description="Provide a detailed description of historical events to answer the question"
    )
    number: str = element(description="A number that is most relevant to the question.")


# Lists of simple types
class ArticleResponse2XML(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="Title of the article")
    answer: str = element(description="Answer the writer's question")
    further_questions: list[str] = element(
        tag="further_question",
        description="A list of related questions that may be of interest to the readers.",
    )


class ListofStrXML(BaseXmlModel):
    """A list of related questions of interest to the readers"""

    further_questions: list[str] = element(
        tag="further_question",
        description="A related question of interest to readers",
    )


# Lists of simple types (encapsulated list)
class ArticleResponse2XMLalt(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="Title of the article")
    answer: str = element(description="Answer the writer's question")
    further_questions: ListofStrXML = element(
        tag="further_questions",
        description="A list of related questions of interest to the readers",
    )


# Nested types
class HistoricalEventXML(BaseXmlModel):
    """The year and explanation of a historical event."""

    year: str = element(description="The year of the historical event")
    event: str = element(
        description="A clear and concise explanation of what happened in this event"
    )


class ArticleResponse3XML(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="[Title of the article]")
    historical_event_1: HistoricalEventXML = element(
        description="A first historical event relevant to the question"
    )
    historical_event_2: HistoricalEventXML = element(
        description="A second historical event relevant to the question"
    )


# Lists of custom types
class ArticleResponse4XML(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="Title of the article")
    historical_timeline: list[HistoricalEventXML] = element(
        description="A list of historical events relevant to the question"
    )


class ListofHistoricalEventXML(BaseXmlModel):
    """A list of historical events relevant to the question"""

    historical_event: list[HistoricalEventXML] = element(
        tag="historical_event",
        description="A relevant historical event",
    )


# Lists of custom types (encapsulated list)
class ArticleResponse4XMLalt(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="Title of the article")
    historical_timeline: ListofHistoricalEventXML = element(
        description="A list of historical events relevant to the question"
    )


## END: Experiment schema definitions
