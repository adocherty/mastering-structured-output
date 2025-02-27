import pickle
import pandas as pd
import xml.etree.ElementTree as ET
from typing import get_origin, get_args, Optional, Any
from pydantic_xml import BaseXmlModel
from langchain_core.language_models import BaseChatModel
from pydantic_xml import BaseXmlModel, element, attr

## START: Experiment schema definitions


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


def load_experiment_summary(suffix: str, date: str, namespace: dict):
    with open(file=f"exp{suffix}_all_models_{date}.pkl", mode="rb") as f:
        data = pickle.load(f)

    # Inject into toplevel namespace
    for key, value in data["models"].items():
        if key not in namespace:
            print(f"Loaded {key}")
            namespace[key] = value


def load_single_experiment(suffix: str, date: str, ident: str, namespace: dict):
    # Load individual model
    with open(
        file=f"exp{suffix}_xml_output_{ident}_{date}.pkl",
        mode="rb",
    ) as f:
        data = pickle.load(f)

    key = f"structure_support_by_model_{ident}"
    if key not in namespace:
        print(f"Loaded {key}")
        namespace[key] = data["structure_support_by_model"]


def pydantic_to_xml_instructions(
    model,
    root_name=None,
    root_description=None,
    add_instructions=True,
    n_list_examples: int = 2,
):
    """This function generates XML schema instructions based on a Pydantic XML model,
    which can be used to guide Large Language Models in producing structured XML output.

    Args:
        model (BaseXmlModel): A Pydantic XML model class
        root_name (str, optional): Custom name for the root XML element.
            Defaults to model's xml_tag or title.
        root_description (str, optional): Custom description for the root element.
            Defaults to the docstring of the class.
        add_instructions (bool, optional): Whether to include prefix instructions
            for the LLM. Defaults to True.
        n_list_examples (int, optional): Number of examples to give for a list (excluding elipsis). Defaults to 2

    Returns:
        str: A string containing XML schema instructions, including:
            - Optional LLM instruction prefix
            - Root element with description
            - Nested elements for each field
            - Type hints and descriptions as XML comments
            - Special handling for lists and nested models

    Example:
        ```python
        from pydantic_xml import BaseXmlModel, element

        class Person(BaseXmlModel):
            name: str = element(description="Person's full name")
            age: int = element(description="Person's age in years")

        format_instructions = pydantic_to_xml_instructions(Person)
        ```
    """
    # Get the JSON schema representation of the model
    schema_json = model.model_json_schema() or {}

    xml = (
        "You must respond only in XML using the following schema.\n"
        "Do not provide any output outside the first and last XML tags.\n\n"
        if add_instructions
        else ""
    )

    # Start with root element named after the model
    _root_name = root_name or model.__xml_tag__ or schema_json.get("title", "")
    _root_desc = root_description or schema_json.get("description", "")
    xml += f"<{_root_name}>\n  <!--{_root_desc}-->\n"

    # Process each property
    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        description = field.description
        tag = field.path if field.path else field_name

        # Handle nested classes
        if isinstance(field_type, type) and issubclass(field_type, BaseXmlModel):
            xml += pydantic_to_xml_instructions(
                field_type, root_name=tag, add_instructions=False
            )
            xml += "\n"

        # Handle lists
        # TODO: lists of lists are not currently handled
        elif get_origin(field_type) is list:
            subtype = get_args(field_type)[0]
            if isinstance(subtype, type) and issubclass(subtype, BaseXmlModel):
                list_xml = pydantic_to_xml_instructions(
                    subtype,
                    root_name=tag,
                    root_description=description,
                    add_instructions=False,
                )
            else:
                list_xml = f"  <{tag}>\n"
                list_xml += f"    {{{description} - must be type {subtype.__name__}}}\n"
                list_xml += f"  </{tag}>"

            # Insert list XML multiple times to prompt a list
            for ii in range(n_list_examples):
                if ii == 1:
                    xml += "<!-- First list element -->\n"
                else:
                    xml += "<!-- Next list element -->\n"
                xml += list_xml + "\n"
            xml += "<!-- Etc -->\n"
            xml += f"  <{tag}>\n  ...\n  </{tag}>\n"

        else:
            # Add field as XML element with type comment and description
            xml += f"  <{tag}>\n"
            xml += f"    {{{description} - must be type {field_type.__name__}}}\n"
            xml += f"  </{tag}>\n"

    xml += f"</{_root_name}>"

    return xml


def extract_substring(input_string, start_str="{", end_str="}"):
    "Extracts the substring enclosed in start_str and end_str"
    start = input_string.find(start_str)
    end = input_string.rfind(end_str)
    if start != -1 and end != -1:
        return input_string[start : end + len(end_str)]
    else:
        raise RuntimeError("ExtractError: End or start strings not found")


class EvalXmlOutput:
    def __init__(self, xml_schema: dict, start_tag="<article>", end_tag="</article>"):
        self._reference_schema = xml_schema
        self._start_tag = start_tag
        self._end_tag = end_tag

    def _calculate_length(self, input_, agg=None):
        if isinstance(input_, int):
            return input_
        elif isinstance(input_, (list, tuple, set)):
            if agg == "sum":
                return sum(self._calculate_length(values) for key, values in input_)
            else:
                return len(input_)
        else:
            return None

    def _parse_xml(self, element, expected_keys):
        structure_ok = True
        errors = []
        schema_output_sizes = []
        for key, value in expected_keys.items():
            child_elements = element.findall(key)
            c_sizes = None

            if len(child_elements) == 0:
                errors.append(f"Missing expected child element: {key}")
                structure_ok = False
                continue

            for child_element in child_elements:
                if child_element is None:
                    errors.append(f"Missing expected child element: {key}")
                    structure_ok = False

                elif isinstance(value, dict):
                    # Nested dictionary, check child elements recursively
                    c_valid, c_sizes, c_errors = self._parse_xml(child_element, value)
                    if not c_valid:
                        errors.extend(c_errors)
                        structure_ok = False

                elif value is None:
                    # Check there are no children of the node
                    children = list(child_element)
                    if len(children) > 0:
                        errors.append(f"Unexpected children for node {key}")
                        structure_ok = False
                    else:
                        c_sizes = len(child_element.text.split())

                schema_output_sizes.append((key, c_sizes))

        return structure_ok, schema_output_sizes, errors

    def __call__(self, xml_string: str) -> dict:
        """Return the validity of the XML schema and sizes of elements"""
        xml_string = extract_substring(
            xml_string, start_str=self._start_tag, end_str=self._end_tag
        )
        try:
            root = ET.fromstring(xml_string)

            xml_valid = True
            xml_schema_ok, output_sizes, errors = self._parse_xml(
                root, self._reference_schema
            )
            xml_schema_reasoning = ". ".join(errors)

        except ET.ParseError:
            xml_valid = False
            xml_schema_ok = False
            output_sizes = dict()
            xml_schema_reasoning = "Error parsing XML"

        except Exception as e:
            xml_valid = False
            xml_schema_ok = False
            output_sizes = dict()
            xml_schema_reasoning = f"Error: {e.__class__.__name__}"

        results = [dict(key="strict_valid", score=xml_valid)]
        results.extend(
            [
                dict(key=key + name, score=score)
                for name, os in output_sizes
                for key, score in [
                    ("len_", self._calculate_length(os, agg=None)),
                ]
            ]
        )
        if xml_schema_ok:
            results.append(dict(key="schema_valid", score=xml_schema_ok))
        else:
            results.append(
                dict(
                    key="f_schema_valid",
                    score=xml_schema_ok,
                    reasoning=xml_schema_reasoning,
                )
            )

        return dict(results=results)


def run_xml_experiment(
    prompt_format,
    questions: list[str],
    llm_models: dict[str, BaseChatModel],
    structured_formats: list[dict[str, Any]],
    n_iter: int = 1,
    resume: int = 0,
    results_out: Optional[dict] = None,
    save_file_name: Optional[str] = None,
):
    """Run XML generation experiments across different models and schema formats.

    This function evaluates how well different language models can generate XML output
    according to specified schemas. It runs multiple iterations across different models
    and formats, tracking success rates and errors.

    Args:
        prompt_format (Template): A prompt template that can be formatted with instructions
        questions (list): List of questions to test XML generation against
        llm_models (dict): Dictionary mapping model names to LLM instances
        structured_formats (list): List of dicts containing 'pydantic' models and
            'format_instructions' for XML generation
        method (str): Identifier for the experiment method being used
        n_iter (int, optional): Number of iterations to run for each question. Defaults to 1
        resume (int, optional): Position to resume from in case of interrupted runs. Defaults to 0
        results_out (dict, optional): Existing results dictionary to append to. Defaults to None
        save_file_name (str, optional): Path to save experiment results. Defaults to None

    Returns:
        dict: Results organized by model and schema, containing:
            - valid: Success rate for XML generation
            - error_types: List of error types encountered
            - errors: Detailed error messages
            - outputs: Raw and parsed outputs for successful generations

    The function saves results to disk if save_file_name is provided, including the method,
    prompt, questions, and structure support data for each model.
    """
    if results_out is None:
        structure_support_by_model = {}
    else:
        structure_support_by_model = results_out
    n_questions = len(questions)

    position = 0

    # Iterate over models
    for model_name, llm_model in llm_models.items():
        if model_name not in structure_support_by_model:
            structure_support_by_model[model_name] = {}

        # Iterate over schemas
        for structure in structured_formats:
            pydantic_obj = structure["pydantic"]

            # Skip over existing experiments
            if pydantic_obj.__name__ in structure_support_by_model[model_name]:
                continue

            # Another way to skip -- deprecate this?
            position += 1
            if position < resume:
                continue

            format_instructions = structure["format_instructions"]
            print(
                f"Model: {model_name}  Output: {pydantic_obj.__name__}   Pos: {position}"
            )

            # Format instructions, if required
            prompt = prompt_format.partial(format_instructions=format_instructions)

            # Iterate over questions
            outputs = []
            output_valid = 0
            for _ in range(n_iter):
                for ii in range(n_questions):
                    parsed = None
                    output = None
                    error_message = None
                    extra_output_chrs = None
                    error_type = "ok"
                    try:
                        test_chain = prompt | llm_model
                        output = test_chain.invoke(dict(question=questions[ii]))

                        # Trim to XML content only
                        start_tag = "<" + pydantic_obj.__xml_tag__ + ">"
                        end_tag = "</" + pydantic_obj.__xml_tag__ + ">"
                        output_xml = extract_substring(
                            output.content, start_tag, end_tag
                        )

                        # Extraneous content
                        extra_output_chrs = len(output.content) - len(output_xml)

                        # Parse the XML
                        parsed = pydantic_obj.from_xml(output_xml)

                        output_valid += 1
                        print(".", end="")

                    # Failures
                    except Exception as e:
                        error_type = "parse_error"
                        # print(f"Error: {type(e).__name__}")
                        error_message = f"{type(e).__name__}, {e}"
                        print("e", end="")

                    finally:
                        outputs.append(
                            dict(
                                raw=output,
                                parsed=parsed,
                                error_type=error_type,
                                error_message=error_message,
                                extra_output_chrs=extra_output_chrs,
                            )
                        )

                    # Pause to avoid timeouts?

                print()

            structure_support_by_model[model_name][pydantic_obj.__name__] = dict(
                valid=output_valid / (n_iter * n_questions),
                outputs=outputs,
            )
    if save_file_name:
        with open(file=save_file_name, mode="wb") as f:
            pickle.dump(
                dict(
                    prompt=prompt_format,
                    questions=questions,
                    structure_support_by_model=structure_support_by_model,
                ),
                f,
            )
    return structure_support_by_model
