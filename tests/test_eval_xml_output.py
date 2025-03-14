import pytest
from experiment_xml import EvalXmlOutput
from pydantic_xml import BaseXmlModel, element


class ExampleXMLSchema(BaseXmlModel, tag="article"):
    """Structured article for publication answering a reader's question."""

    title: str = element(description="Title of the article")
    answer: str = element(description="Answer the writer's question")
    further_questions: list[str] = element(
        tag="further_question",
        description="A list of related questions that may be of interest to the readers.",
    )
