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
    base_model = getattr(self.model, "module", self.model)  # unwrap DDP if needed
    base_model.eval()

    original_config = base_model.config.output_attentions
    base_model.config.output_attentions = True

    with torch.no_grad():
        decoder_input_ids = None
        if base_model.config.is_encoder_decoder:
            decoder_input_ids = torch.full(
                (input_ids.size(0), 1),
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else base_model.config.decoder_start_token_id,
                dtype=torch.long,
                device=input_ids.device
            )

        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True,
        )

    base_model.config.output_attentions = original_config

    if base_model.config.is_encoder_decoder:
        attentions = outputs.cross_attentions  # Input-focused attention
    else:
        attentions = outputs.attentions

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
