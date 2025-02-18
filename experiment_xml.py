import pickle
import xml.etree.ElementTree as ET
from typing import get_origin, get_args
from pydantic_xml import BaseXmlModel


def pydantic_to_xml_instructions(
    model, root_name=None, root_description=None, add_instructions=True
):
    """Converts a Pydantic XML model to XML format instructions"""

    # Get the JSON schema representation of the model
    schema_json = model.model_json_schema() or {}

    xml = (
        "You must respond only in XML using the following schema:\n"
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

        # Handle lists (but not lists of lists!)
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
                list_xml += f"    {description}\n"
                list_xml += f"  </{tag}>"

            xml += "<!-- First list element -->\n"
            xml += list_xml + "\n"
            xml += "<!-- Next list element -->\n"
            xml += list_xml + "\n"
            xml += "<!-- Etc -->\n"
            xml += f"  <{tag}>\n  ...\n  </{tag}>\n"

        else:
            # Add field as XML element with type comment and description
            xml += f"  <{tag}>\n"
            xml += f"    {{{description}}}\n"
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
        return input_string


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

        except ET.ParseError as e:
            xml_valid = False
            xml_schema_ok = False
            output_sizes = dict()
            xml_schema_reasoning = "Error parsing XML"

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
    questions,
    llm_models,
    structured_formats,
    method,
    n_iter=1,
    resume=0,
    results_out=None,
    save_file_name=None,
):

    if results_out is None:
        structure_support_by_model = {}
    else:
        structure_support_by_model = results_out
    n_questions = len(questions)

    position = 0

    # Iterate over models
    for model_name, llm_model in llm_models.items():
        structure_support_by_model[model_name] = {}

        # Iterate over schemas
        for structure in structured_formats:
            pydantic_obj = structure["pydantic"]
            format_instructions = structure["format_instructions"]
            print(
                f"Model: {model_name}  Output: {pydantic_obj.__name__}   Pos: {position}"
            )

            position += 1
            if position < resume:
                continue

            # Format instructions, if required
            prompt = prompt_format.partial(format_instructions=format_instructions)

            # Iterate over questions
            error_types = []
            error_messages = []
            outputs = []
            output_valid = 0
            for _ in range(n_iter):
                for ii in range(n_questions):
                    try:
                        test_chain = prompt | llm_model
                        output = test_chain.invoke(dict(question=questions[ii]))

                        # Parse the XML
                        parsed = pydantic_obj.from_xml(output)
                        outputs.append(dict(raw=output.content, parsed=parsed))

                        error_types.append("ok")
                        output_valid += 1

                    # Failures
                    except Exception as e:
                        error_types.append("parse_error")
                        print(f"Error: {type(e).__name__}")
                        error_messages.append(f"{type(e).__name__}, {e}")

                    # Pause to avoid timeouts
                    print(".", end="")
                print()

            structure_support_by_model[model_name][pydantic_obj.__name__] = dict(
                valid=output_valid / (n_iter * n_questions),
                error_types=error_types,
                errors=error_messages,
                outputs=outputs,
            )
    if save_file_name:
        with open(file=save_file_name, mode="wb") as f:
            pickle.dump(
                dict(
                    method=method,
                    prompt=prompt,
                    questions=questions,
                    structure_support_by_model=structure_support_by_model,
                ),
                f,
            )
    return structure_support_by_model
