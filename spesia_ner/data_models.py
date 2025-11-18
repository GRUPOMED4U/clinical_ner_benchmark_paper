from pathlib import Path
from pydantic import BaseModel
from typing import List, Literal, Set, Union
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
    """
    Representa uma anotação de texto com posições, tags e grupos semânticos.
    """

    id: str
    tags: Set[str] = set()
    semantic_groups: Set[str] = set()
    start: int
    end: int
    text: str


class Record(BaseModel):
    """
    Representa um registro de texto anotado, contendo o texto completo
    e suas anotações (entidades, rótulos, etc.).
    """

    text: str
    annotations: List[Annotation]

    # ==============================================================
    #  MÉTODO 1 - XML (SemClinBr)
    # ==============================================================

    @classmethod
    def from_xml(cls, file_path: Path | str) -> "Record":
        """
        Lê um arquivo XML no formato SemClinBr e retorna um Record.
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

    # ==============================================================
    #  MÉTODO 2 - JSON (Argilla)
    # ==============================================================

    @classmethod
    def from_argilla_json(cls, file_path: Path) -> list["Record"]:
        """
        Lê um arquivo JSON exportado do Argilla (formato dict com IDs como chaves).
        Exemplo de estrutura:
        {
        "<uuid>": {
            "fields": {"text": "..."},
            "responses": {
            "annotations": [
                {"value": [{"label": "TAG", "start": 10, "end": 20}]}
            ]
            }
        }
        }
        """
        import json

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for record_id, record_data in data.items():
            # segurança: só processa se tiver 'fields' e 'responses'
            if not isinstance(record_data, dict):
                continue

            fields = record_data.get("fields", {})
            text = fields.get("text", "").strip()

            # tenta localizar as anotações (em responses.annotations)
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

    # ==============================================================
    #  MÉTODO 3 - JSON/JSONL (Docanno)
    # ==============================================================

    @classmethod
    def from_doccano_jsonl(cls, file_path: Path) -> list["Record"]:
        """
        Lê um arquivo JSONL exportado do Doccano.
        Suporta diferentes variantes de export:
        - {"text": "...", "labels": [[start, end, "TAG"], ...]}
        - {"text": "...", "label": [[start, end, "TAG"], ...]}
        - {"text": "...", "entities": [{"start":..,"end":..,"label":..}, ...]}
        - {"text": "...", "annotations": [{"start_offset":..,"end_offset":..,"label":..}, ...]}
        """
        import json

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
                    # 1. Caso lista [start, end, label]
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        start, end, tag = item[:3]
                        # Fix index offset
                        start -= 1
                        end -= 1
                    # 2. Caso dict com start/end/label
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

    # ==============================================================
    #  MÉTODO 4 - DETECÇÃO AUTOMÁTICA
    # ==============================================================

    @classmethod
    def from_file(cls, file_path: Path | str) -> list["Record"]:
        """
        Detecta automaticamente o formato de arquivo (XML, JSON Argilla ou JSONL Doccano)
        e chama o parser apropriado.
        """
        import json

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
                # Dicionário Argilla: chaves UUID → registros
                if isinstance(data, dict) and "fields" in next(iter(data.values())):
                    return cls.from_argilla_json(file_path)
                # Lista Doccano JSON (pouco comum)
                elif isinstance(data, list) and any("text" in d for d in data):
                    # converte lista simples tipo [{"text": ..., "labels": [...]}]
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
                    raise ValueError("Formato JSON não reconhecido.")
            except Exception as e:
                print(f"[WARN] Erro ao processar {file_path.name}: {e}")
                return []

        # --- JSONL ---
        elif suffix == ".jsonl":
            return cls.from_doccano_jsonl(file_path)

        else:
            raise ValueError(f"Formato de arquivo não suportado: {suffix}")

    # ==============================================================
    #  PROPRIEDADES AUXILIARES
    # ==============================================================

    @property
    def tags(self) -> Set[str]:
        """
        Retorna o conjunto de todas as tags contidas nas anotações.
        """
        tags = set()
        for annotation in self.annotations:
            tags.update(annotation.tags)
        return tags

    @property
    def semantic_groups(self) -> Set[str]:
        """
        Retorna o conjunto de todos os grupos semânticos contidos nas anotações.
        """
        semantic_groups = set()
        for annotation in self.annotations:
            semantic_groups.update(annotation.semantic_groups)
        return semantic_groups

    def to_agent_annotations(
        self, label_type: Literal["tags", "semantic_groups"]
    ) -> AgentAnnotationsList:
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
