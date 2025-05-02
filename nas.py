# nas.py
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm  # Added tqdm import for progress tracking

class NegativeAttentionScorer:
    """
    NAS measures the tendency of attention heads to attend to
    negative answer tokens ("Not", "Never", "No") based on which appears last in the sentence.
    """
    
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase,
        output_attentions: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_attentions = output_attentions

        self.not_token_ids = self.tokenizer.encode("not", add_special_tokens=False)
        self.never_token_ids = self.tokenizer.encode("never", add_special_tokens=False)
        self.no_token_ids = self.tokenizer.encode("no", add_special_tokens=False)

        self.not_token_id = self.not_token_ids[-1]
        self.never_token_id = self.never_token_ids[-1]
        self.no_token_id = self.no_token_ids[-1]

        logging.info(f"not token ID: {self.not_token_id}")
        logging.info(f"never token ID: {self.never_token_id}")
        logging.info(f"no token ID: {self.no_token_id}")

    def calculate_nas(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        pattern_ids: List[int],
        negation_types: List[str],
        test_ids: List[int],
        labels: List[str],
        sentences: List[str]
    ) -> List[Dict[str, Union[float, List[List[float]]]]]:
        """
        Calculate Negative Attention Score for input examples.
        """
        self.model.eval()
        original_config = self.model.config.output_attentions
        self.model.config.output_attentions = True

        with torch.no_grad():
            decoder_input_ids = torch.full(
                (input_ids.size(0), 1),
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.model.config.decoder_start_token_id,
                dtype=torch.long,
                device=input_ids.device
            )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                output_attentions=True
            )


        self.model.config.output_attentions = original_config
        attentions = outputs.cross_attentions
        batch_size = input_ids.shape[0]
        results = []
        for batch_idx in range(batch_size):
            pattern_id = pattern_ids[batch_idx]
            neg_type = negation_types[batch_idx]

            if neg_type == "affirmation":
                continue

            sentence = sentences[batch_idx].lower()
            last_positions = {
                "not": sentence.rfind("not "),
                "never": sentence.rfind("never "),
                "no": sentence.rfind("no ")
            }
            last_token = max(last_positions, key=last_positions.get)
            if last_positions[last_token] == -1:
                logging.warning(
                    f"Could not find any negation token in sample {batch_idx} "
                    f"(test_id: {test_ids[batch_idx]}, sentence: {sentences[batch_idx]})"
                )
                continue

            if last_token == "not":
                target_token_id = self.not_token_id
            elif last_token == "never":
                target_token_id = self.never_token_id
            elif last_token == "no":
                target_token_id = self.no_token_id

            sample_input_ids = input_ids[batch_idx]
            sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_input_ids.tolist())

            token_positions = (sample_input_ids == target_token_id).nonzero(as_tuple=True)[0]

            if len(token_positions) == 0:
                token_positions = torch.tensor([
                    i for i, token in enumerate(sample_tokens)
                    if self.tokenizer.decode([target_token_id]).strip().lower() in token.lower()
                ])

            if len(token_positions) == 0:
                logging.warning(
                    f"Could not find target token '{last_token}' in sample {batch_idx} "
                    f"(test_id: {test_ids[batch_idx]}, sentence: {sentences[batch_idx]})"
                )
                continue

            target_pos = token_positions[-1].item()
            all_nas_scores = []

            for layer_attentions in attentions:
                layer_nas = []
                for head_idx in range(layer_attentions.shape[1]):
                    head_attention = layer_attentions[batch_idx, head_idx]
                    attn_score = head_attention[:, target_pos].sum()
                    layer_nas.append(attn_score.item())
                all_nas_scores.append(layer_nas)

            flat_nas = [score for layer in all_nas_scores for score in layer]
            single_head_nas = sum(flat_nas) / len(flat_nas) if flat_nas else 0.0
            model_nas = sum(flat_nas) if flat_nas else 0.0

            results.append({
                "test_id": test_ids[batch_idx],
                "label": labels[batch_idx],
                "sentence": sentences[batch_idx],
                "last_token": last_token,
                "single_head_nas": single_head_nas,
                "model_nas": model_nas,
                "pattern_id": pattern_id,
                "negation_type": neg_type
            })

        return results

def run_nas_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataloader,
    output_path: str,
    accelerator,
):
    """
    Run NAS evaluation on the given dataloader and save results to output_path.
    """
    scorer = NegativeAttentionScorer(model, tokenizer)

    all_results = []
    sample_count = 0

    model.eval()
    with torch.no_grad():
        # Add progress bar for NAS evaluation
        for step, batch in enumerate(tqdm(dataloader, desc="NAS Evaluation")):
            jsonl_data = dataloader.dataset.get_jsonl()
            examples = jsonl_data[sample_count:sample_count + len(batch["input_ids"])]
            pattern_ids = [ex["pattern_id"] for ex in examples]
            negation_types = [ex["negation_type"] for ex in examples]
            test_ids = [ex["test_id"] for ex in examples]
            labels = [ex["label"] for ex in examples]
            sentences = [ex["sentence"] for ex in examples]

            nas_results = scorer.calculate_nas(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pattern_ids=pattern_ids,
                negation_types=negation_types,
                test_ids=test_ids,
                labels=labels,
                sentences=sentences
            )

            sample_count += len(batch["input_ids"])
            all_results.extend(nas_results)

    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)

    logging.info(f"NAS evaluation results saved to {output_path}")
    return df
