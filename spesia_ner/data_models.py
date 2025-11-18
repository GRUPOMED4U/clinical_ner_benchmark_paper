from pathlib import Path
from pydantic import BaseModel
from typing import List, Literal, Set
import xml.etree.ElementTree as ET
import json


class AgentAnnotation(BaseModel):
    label: str
    text: str


class AgentAnnotationsList(BaseModel):
    annotations: List[AgentAnnotation]

    def to_record(
        self, text: str, label_type: Literal["tags", "semantic_groups"]
    ) -> "Record":
        return Record.from_agent_annotations(text, self, label_type)


class AgentGeneratedRecord(BaseModel):
    text: str
    annotations: List[AgentAnnotation]

    def to_record(
        self, label_type: Literal["tags", "semantic_groups"] = "tags"
    ) -> "Record":
        return Record.from_agent_annotations(self.text, self, label_type)


class Annotation(BaseModel):
    id: str
    tags: Set[str] = set()
    semantic_groups: Set[str] = set()
    start: int
    end: int
    text: str


class Record(BaseModel):
    text: str
    annotations: List[Annotation]

    @classmethod
    def from_xml(cls, file_path: Path | str) -> "Record":
        """
        Parse an XML file into a Record object.

        Args:
            file_path (Path | str): Path to the XML file.

        Returns:
            Record: A Record object containing the text and annotations from the XML file.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        tree = ET.parse(file_path)
        root = tree.getroot()

        text = root.find("TEXT").text
        tags = root.find("TAGS")
        annotations = []

        for tag in tags:
            current_annotation = Annotation(
                id=tag.attrib["id"],
                tags=set(tag.attrib["tag"].split("|")),
                start=int(tag.attrib["start"]),
                end=int(tag.attrib["end"]),
                text=text[int(tag.attrib["start"]) : int(tag.attrib["end"])],
            )
            annotations.append(current_annotation)

        return cls(text=text, annotations=annotations)

    @classmethod
    def from_argilla_json(cls, file_path: Path) -> list["Record"]:
        """
        Parse an Argilla JSON file into a list of Record objects.

        Args:
            file_path (Path): Path to the Argilla JSON file.

        Returns:
            list[Record]: A list of Record objects containing the text and annotations from the Argilla JSON file.

        Notes:
            - The function assumes that the JSON file is in the same format as the one exported by the Argilla annotation tool.
            - The function processes each block of annotations by one user as a separate record.
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for record_id, record_data in data.items():
            # skip non-dict records
            if not isinstance(record_data, dict):
                continue

            fields = record_data.get("fields", {})
            text = fields.get("text", "").strip()

            responses = record_data.get("responses", {})
            annotations_per_user = responses.get("annotations", [])

            for user_annotations in annotations_per_user:
                # each block contains annotations by one user
                # create a different record for each
                annotations = []
                for ann in user_annotations.get("value", []):
                    start = ann.get("start")
                    end = ann.get("end")
                    label = ann.get("label")

                    if start is None or end is None or not label:
                        continue

                    annotations.append(
                        Annotation(
                            id=record_id,
                            tags={label},
                            start=start,
                            end=end,
                            text=text[start:end],
                        )
                    )

                records.append(cls(text=text, annotations=annotations))

        return records

    @classmethod
    def from_doccano_jsonl(cls, file_path: Path) -> list["Record"]:
        """
        Parse a Doccano JSONL file into a list of Record objects.

        Args:
            file_path (Path): Path to the Doccano JSONL file.

        Returns:
            list[Record]: A list of Record objects containing the text and annotations from the Doccano JSONL file.

        Notes:
            - The function assumes that the JSONL file is in the same format as the one exported by the Doccano annotation tool.
            - The function processes each block of annotations by one user as a separate record.
        """
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                entry = json.loads(line)
                text = entry.get("text", "").strip()

                annotations = []
                raw_labels = (
                    entry.get("labels")
                    or entry.get("label")
                    or entry.get("entities")
                    or entry.get("annotations")
                    or []
                )

                for i, item in enumerate(raw_labels):
                    # 1. List case [start, end, label]
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        start, end, tag = item[:3]

                    # 2. Dict case start/end/label
                    elif isinstance(item, dict):
                        start = (
                            item.get("start")
                            or item.get("start_offset")
                            or item.get("begin")
                        )
                        end = (
                            item.get("end")
                            or item.get("end_offset")
                            or item.get("stop")
                        )
                        tag = item.get("label") or item.get("labels")
                    else:
                        continue

                    if start is None or end is None or not tag:
                        continue

                    annotations.append(
                        Annotation(
                            id=str(i),
                            tags={tag} if isinstance(tag, str) else set(tag),
                            start=int(start),
                            end=int(end),
                            text=text[int(start) : int(end)],
                        )
                    )

                records.append(cls(text=text, annotations=annotations))

        return records

    @classmethod
    def from_file(cls, file_path: Path | str) -> list["Record"]:
        """
        Parse a file containing annotations into a list of Record objects.

        Supports the following formats:
        - XML (Argilla format)
        - JSON (Argilla format and Doccano JSON format)
        - JSONL (Doccano JSONL format)

        Args:
            file_path (Path | str): Path to the file containing the annotations.

        Returns:
            list[Record]: A list of Record objects containing the text and annotations from the file.

        Raises:
            ValueError: If the file format is not recognized.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        suffix = file_path.suffix.lower()

        # --- XML ---
        if suffix == ".xml":
            return [cls.from_xml(file_path)]

        # --- JSON ---
        elif suffix == ".json":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict) and "fields" in next(iter(data.values())):
                    return cls.from_argilla_json(file_path)

                elif isinstance(data, list) and any("text" in d for d in data):
                    return [
                        cls(
                            text=item["text"],
                            annotations=[
                                Annotation(
                                    id=str(i),
                                    tags=(
                                        {lbl[2]}
                                        if isinstance(lbl, list)
                                        else {lbl.get("label")}
                                    ),
                                    start=(
                                        lbl[0]
                                        if isinstance(lbl, list)
                                        else lbl.get("start")
                                    ),
                                    end=(
                                        lbl[1]
                                        if isinstance(lbl, list)
                                        else lbl.get("end")
                                    ),
                                    text=(
                                        item["text"][lbl[0] : lbl[1]]
                                        if isinstance(lbl, list)
                                        else item["text"][lbl["start"] : lbl["end"]]
                                    ),
                                )
                                for i, lbl in enumerate(
                                    item.get("labels", []) or item.get("label", [])
                                )
                            ],
                        )
                        for item in data
                    ]
                else:
                    raise ValueError("Unkown JSON pattern.")
            except Exception as e:
                print(f"[WARN] Error while processing {file_path.name}: {e}")
                return []

        # --- JSONL ---
        elif suffix == ".jsonl":
            return cls.from_doccano_jsonl(file_path)

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @property
    def tags(self) -> Set[str]:
        """
        Returns a set of all tags in the record's annotations.

        :return: Set[str]
        """
        tags = set()
        for annotation in self.annotations:
            tags.update(annotation.tags)
        return tags

    @property
    def semantic_groups(self) -> Set[str]:
        """
        Returns a set of all semantic groups in the record's annotations.

        :return: Set[str]
        """
        semantic_groups = set()
        for annotation in self.annotations:
            semantic_groups.update(annotation.semantic_groups)
        return semantic_groups

    def to_agent_annotations(
        self, label_type: Literal["tags", "semantic_groups"]
    ) -> AgentAnnotationsList:
        """
        Converts the record's annotations to AgentAnnotationsList.

        Args:
            label_type (Literal["tags", "semantic_groups"]): The type of label to extract from the annotations.

        Returns:
            AgentAnnotationsList: A list of AgentAnnotation objects.
        """
        agent_annotations = AgentAnnotationsList(annotations=[])
        for annotation in self.annotations:
            for label in getattr(annotation, label_type):
                agent_annotations.annotations.append(
                    AgentAnnotation(label=label, text=annotation.text)
                )
        return agent_annotations

    @classmethod
    def from_agent_annotations(
        cls,
        text: str,
        agent_annotations: AgentAnnotationsList,
        label_type: Literal["tags", "semantic_groups"],
    ) -> "Record":
        """
        Converts an AgentAnnotationsList to a Record.

        Args:
            text (str): The text associated with the record.
            agent_annotations (AgentAnnotationsList): The list of AgentAnnotation objects to convert.
            label_type (Literal["tags", "semantic_groups"]): The type of label to extract from the annotations.

        Returns:
            Record: A new Record object with the converted annotations.
        """
        new_record = cls(text=text, annotations=[])

        for i, annotation in enumerate(agent_annotations.annotations):
            start_index = text.find(annotation.text)
            if start_index != -1:
                new_annotation = Annotation(
                    id=str(i),
                    tags=set([annotation.label]) if label_type == "tags" else set(),
                    semantic_groups=(
                        set([annotation.label])
                        if label_type == "semantic_groups"
                        else set()
                    ),
                    start=start_index,
                    end=start_index + len(annotation.text),
                    text=annotation.text,
                )
                new_record.annotations.append(new_annotation)
        return new_record
