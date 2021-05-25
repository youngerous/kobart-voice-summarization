import warnings
from typing import List

import torch.nn as nn
from transformers import BartForConditionalGeneration

LAYERS_TO_COPY = {
    6: {
        1: [0],
        2: [0, 5],
        3: [0, 2, 5],
        4: [0, 1, 3, 5],
        6: [0, 1, 2, 3, 4, 5],
    },
}


class DistilBART(nn.Module):
    """
    Distilled version using Shrink-and-Finetune(SFT) method.

    Args:
        teacher (BartForConditionalGeneration): Teacher model to distill
        n_enc (int): Number of student encoder layers
        n_dec (int): Number of student decoder layers

    Papers:
        Pre-trained Summarization Distillation(https://arxiv.org/abs/2010.13002)

    Code Reference:
        https://github.com/huggingface/transformers/blob/49e4fece5c5cfb31615a3bddcff15517333e6fb6/examples/seq2seq/make_student.py
    """

    def __init__(self, teacher, n_enc=None, n_dec=None):
        super(DistilBART, self).__init__()
        # load teacher
        self.teacher = teacher

        # set student config
        init_kwargs = self.teacher.config.to_diff_dict()
        teacher_e, teacher_d = (
            self.teacher.config.encoder_layers,
            self.teacher.config.decoder_layers,
        )
        if n_enc is None:
            n_enc = teacher_e
        if n_dec is None:
            n_dec = teacher_d
        init_kwargs.update({"encoder_layers": n_enc, "decoder_layers": n_dec})

        # init student
        student_cfg = self.teacher.config_class(**init_kwargs)
        self.student = BartForConditionalGeneration(config=student_cfg)
        self.student.load_state_dict(teacher.state_dict(), strict=False)

        e_layers_to_copy: List[int] = self.pick_layers_to_copy(n_enc, teacher_e)
        d_layers_to_copy: List[int] = self.pick_layers_to_copy(n_dec, teacher_d)

        # copy layer weight
        self.copy_layers(
            self.teacher.model.encoder.layers,
            self.student.model.encoder.layers,
            e_layers_to_copy,
        )
        self.copy_layers(
            self.teacher.model.decoder.layers,
            self.student.model.decoder.layers,
            d_layers_to_copy,
        )

        print(
            f"[Teacher] Encoder {len(self.teacher.model.encoder.layers)}-layer / Decoder {len(self.teacher.model.decoder.layers)}-layer"
        )
        print(
            f"[Student] Encoder {len(self.student.model.encoder.layers)}-layer / Decoder {len(self.student.model.decoder.layers)}-layer"
        )
        self.teacher = None

    def forward(self, input_ids, attention_mask, labels):
        return self.student(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def copy_layers(
        self,
        src_layers: nn.ModuleList,
        dest_layers: nn.ModuleList,
        layers_to_copy: List[int],
    ) -> None:
        layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
        assert len(dest_layers) == len(
            layers_to_copy
        ), f"{len(dest_layers)} != {len(layers_to_copy)}"
        dest_layers.load_state_dict(layers_to_copy.state_dict())

    def pick_layers_to_copy(self, n_student: int, n_teacher: int):
        try:
            val = LAYERS_TO_COPY[n_teacher][n_student]
            return val
        except KeyError:
            if n_student != n_teacher:
                warnings.warn(
                    f"no hardcoded layers to copy for teacher {n_teacher} -> student {n_student}, defaulting to first {n_student}"
                )
            return list(range(n_student))
