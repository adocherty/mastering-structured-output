from pydantic_xml.element import SearchMode
from pydantic_xml import BaseXmlModel, element, attr
from typing import Dict, Optional
import copyreg
import pickle

__all__ = [
    "generate_xml_classes",
    "DynamicPXUnpickler",
    "ArticleResponse1XML",
    "ArticleResponse1nointXML",
    "ArticleResponse2XML",
    "ArticleResponse2XMLalt",
    "ArticleResponse3XML",
    "ArticleResponse4XML",
    "ArticleResponse4XMLalt",
    "ListofStrXML",
    "ListofHistoricalEventXML",
    "HistoricalEventXML",
]

DEFAULT_SEARCH_MODE = SearchMode.STRICT

## START: XML schema definitions


class ArticleResponse1XML(BaseXmlModel, tag="article", search_mode=DEFAULT_SEARCH_MODE):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="Title of the article")
    answer: str = element(
        description="Provide a detailed description of historical events to answer the question"
    )
    number: int = element(description="A number that is most relevant to the question.")


class ArticleResponse1nointXML(
    BaseXmlModel, tag="article", search_mode=DEFAULT_SEARCH_MODE
):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="Title of the article")
    answer: str = element(
        description="Provide a detailed description of historical events to answer the question"
    )
    number: str = element(description="A number that is most relevant to the question.")


# Lists of simple types
class ArticleResponse2XML(BaseXmlModel, tag="article", search_mode=DEFAULT_SEARCH_MODE):
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
class ArticleResponse2XMLalt(
    BaseXmlModel, tag="article", search_mode=DEFAULT_SEARCH_MODE
):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="Title of the article")
    answer: str = element(description="Answer the writer's question")
    further_questions: ListofStrXML = element(
        tag="further_questions",
        description="A list of related questions of interest to the readers",
    )


# Nested types
class HistoricalEventXML(BaseXmlModel, search_mode=DEFAULT_SEARCH_MODE):
    """The year and explanation of a historical event."""

    year: str = element(description="The year of the historical event")
    event: str = element(
        description="A clear and concise explanation of what happened in this event"
    )


class ArticleResponse3XML(BaseXmlModel, tag="article", search_mode=DEFAULT_SEARCH_MODE):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="[Title of the article]")
    historical_event_1: HistoricalEventXML = element(
        description="A first historical event relevant to the question"
    )
    historical_event_2: HistoricalEventXML = element(
        description="A second historical event relevant to the question"
    )


# Lists of custom types
class ArticleResponse4XML(BaseXmlModel, tag="article", search_mode=DEFAULT_SEARCH_MODE):
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
class ArticleResponse4XMLalt(
    BaseXmlModel, tag="article", search_mode=DEFAULT_SEARCH_MODE
):
    """Structured article for publication answering a reader's question"""

    title: str = element(description="Title of the article")
    historical_timeline: ListofHistoricalEventXML = element(
        description="A list of historical events relevant to the question"
    )


## END: Experiment schema definitions


class DynamicPXUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        self._search_mode = kwargs.pop("search_mode", None)
        self._classes = generate_xml_classes(self._search_mode)
        return super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == "pydantic_structure_definitions":
            return self._classes.get(name, globals().get(name))
        return super().find_class(module, name)


def generate_xml_classes(search_mode: Optional[str] = None):
    # Create new class variants with specified search_mode
    class ArticleResponse1XML_Dynamic(ArticleResponse1XML, search_mode=search_mode):
        pass

    class ArticleResponse1nointXML_Dynamic(
        ArticleResponse1nointXML, search_mode=search_mode
    ):
        pass

    class ArticleResponse2XML_Dynamic(ArticleResponse2XML, search_mode=search_mode):
        pass

    class ArticleResponse3XML_Dynamic(ArticleResponse3XML, search_mode=search_mode):
        pass

    class ArticleResponse4XML_Dynamic(ArticleResponse4XML, search_mode=search_mode):
        pass

    class ArticleResponse2XMLalt_Dynamic(
        ArticleResponse2XMLalt, search_mode=search_mode
    ):
        pass

    class ArticleResponse4XMLalt_Dynamic(
        ArticleResponse4XMLalt, search_mode=search_mode
    ):
        pass

    class HistoricalEventXML_Dynamic(HistoricalEventXML, search_mode=search_mode):
        pass

    classes = {}
    for dname, value in locals().items():
        if dname.endswith("Dynamic"):
            new_name = dname.removesuffix("_Dynamic")
            value.__name__ = new_name  # Need to update name as it's used in code to identify the class
            classes[new_name] = value
    return classes


class DynamicPXUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        self._search_mode = kwargs.pop("search_mode", None)
        super().__init__(*args, **kwargs)
        self._classes = generate_xml_classes(self._search_mode)

    def find_class(self, module, name):
        if module == "pydantic_structure_definitions":
            return self._classes.get(name, globals().get(name))
        return super().find_class(module, name)
