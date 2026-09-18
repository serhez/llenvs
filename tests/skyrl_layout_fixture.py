"""Shared native layout fixture; no native imports until explicitly called."""


def layout_batch(tokenizer, vocabulary, estimator, weighting):
    import torch
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
    from skyrl.backends.skyrl_train.utils.ppo_utils import compute_grpo_outcome_advantage
    from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors

    from llenvs.integrations.skyrl._credit import compute_credit
    from llenvs.integrations.skyrl._rendering import TextRenderer
    from tests.test_skyrl_credit import row

    renderer = TextRenderer(tokenizer, vocab_size=vocabulary, chat_template_kwargs={})
    prompts, responses, masks, rewards, ledger = [], [], [], [], []
    for index in range(4):
        prompt = renderer.initial([{"role": "user", "content": "Task " + "detail " * (index + 1)}])
        response, mask, reward, spans = [], [], [], []
        for turn in range(1 + index % 2):
            if turn:
                extended = renderer.extend(
                    prompt + response, {"role": "user", "content": "Observation " * (index + 1)}
                )
                gap = extended[len(prompt) + len(response) :]
                response.extend(gap)
                mask.extend([0] * len(gap))
                reward.extend([0.0] * len(gap))
            tokens = tokenizer.encode("answer " * (index + turn + 1), add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]
            start = len(response)
            response.extend(tokens)
            mask.extend([1] * len(tokens))
            immediate = [0.0] * len(tokens)
            if estimator == "llenvs_token_rtg":
                immediate = [(token % 3 - 1) / 10 for token in tokens]
            immediate[-1] += (index % 2 + 1) * (turn + 1)
            reward.extend(immediate)
            spans.append((start, len(response)))
        if estimator == "grpo":
            reward = [0.0] * (len(response) - 1) + [sum(reward)]
        prompts.append(prompt)
        responses.append(response)
        masks.append(mask)
        rewards.append(reward)
        ledger.append(row(f"group-{index // 2}", index % 2, len(response), spans))
    sequences, attention, response_mask, reward_tensor, loss_mask, _, _ = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer.pad_token_id,
            prompts,
            responses,
            rewards,
            masks,
            max_seq_len=1,
        )
    )
    if estimator == "grpo":
        advantages, returns = compute_grpo_outcome_advantage(
            reward_tensor, loss_mask, [r["instance_id"] for r in ledger], grpo_norm_by_std=True
        )
    else:
        advantages, returns = compute_credit(
            reward_tensor,
            attribution=ledger,
            estimator=estimator,
            n_samples_per_prompt=2,
            grpo_norm_by_std=estimator == "llenvs_turn_grpo",
            turn_weighting=weighting,
        )
    batch = TrainingInputBatch(
        dict(
            sequences=sequences,
            attention_mask=attention,
            response_mask=response_mask,
            rewards=reward_tensor,
            advantages=advantages,
            returns=returns,
            loss_mask=loss_mask,
            row_ids=torch.arange(4),
        )
    )
    batch.metadata = {"response_length": loss_mask.shape[1]}
    return batch, dict(prompts=prompts, responses=responses, sampled_spans=ledger, loss_masks=masks)
