//! End-to-end tests against the *public* API surface.
//!
//! These live outside the crate on purpose: they can only use what a real
//! consumer can reach, which is what makes them a compatibility guard for the
//! library target rather than another internal unit test.

use noetic::autograd::Graph;
use noetic::bpe::Bpe;
use noetic::ckpt;
use noetic::data::{synthetic_corpus, Batcher};
use noetic::infer::{sample_token, Decoder, SampleCfg};
use noetic::model::{Lm, LmConfig};
use noetic::optim::{AdamW, Lion, Schedule};
use noetic::rng::Rng;

/// A deliberately tiny model, derived from the library's own preset so that a
/// change to `LmConfig::small` shows up here instead of silently diverging.
fn tiny_config(vocab: usize) -> LmConfig {
    LmConfig { d_model: 24, conv_k: 3, mlp_mult: 2, tau_max: 16.0, ..LmConfig::small(vocab) }
}

fn unique_path(tag: &str) -> String {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|elapsed| elapsed.as_nanos()).unwrap_or(0);
    std::env::temp_dir().join(format!("noetic-it-{}-{}-{}", tag, std::process::id(), nanos)).to_string_lossy().into_owned()
}

/// Tokenize a corpus, train briefly, checkpoint, reload into a fresh graph and
/// confirm the restored model streams identical logits.
#[test]
fn train_checkpoint_reload_reproduces_logits() {
    let corpus = synthetic_corpus(20_000, 11);
    let tokenizer = Bpe::train(&corpus, 320, false);
    let tokens = tokenizer.encode(&corpus);
    assert!(tokens.len() > 512, "corpus should tokenize to a usable stream");

    let cfg = tiny_config(tokenizer.vocab_size());
    let batcher = Batcher::new(tokens, 0.05);
    let mut rng = Rng::new(4);
    let mut graph = Graph::new(2);
    let model = Lm::new(&mut graph, &mut rng, cfg);
    graph.seal_params();
    let mut optimizer = AdamW::new(&graph, 0.01);
    let schedule = Schedule { peak: 3e-3, min: 3e-4, warmup: 2, total: 30 };

    let mut first = 0.0f32;
    let mut last = 0.0f32;
    for step in 0..30 {
        graph.reset();
        graph.zero_grad();
        let (x, y) = batcher.train_batch(4, 32, &mut rng);
        let (_, loss) = model.loss(&mut graph, &x, &y, 4, 32);
        graph.backward(loss);
        graph.clip_grad_norm(1.0);
        optimizer.step(&mut graph, schedule.lr(step));
        last = graph.scalar(loss);
        if step == 0 {
            first = last;
        }
    }
    assert!(last < first, "training should reduce the loss ({} -> {})", first, last);

    let path = unique_path("ckpt");
    let meta = vec![("vocab".to_string(), cfg.vocab.to_string())];
    ckpt::save(&path, &graph, &meta).expect("checkpoint save");

    let mut restored_graph = Graph::new(2);
    let mut other_rng = Rng::new(9999);
    let restored = Lm::new(&mut restored_graph, &mut other_rng, cfg);
    restored_graph.seal_params();
    let loaded = ckpt::load(&path).expect("checkpoint load");
    let applied = ckpt::apply_exact(&mut restored_graph, &loaded).expect("checkpoint apply");
    assert_eq!(applied, restored_graph.params.len());
    let _ = std::fs::remove_file(&path);

    let ids: Vec<u32> = (0..6).map(|i| ((i * 5 + 1) % cfg.vocab) as u32).collect();
    let mut original_decoder = Decoder::new(&cfg);
    let mut restored_decoder = Decoder::new(&cfg);
    for &id in &ids {
        let expected = original_decoder.step(&graph, &model, id).to_vec();
        let actual = restored_decoder.step(&restored_graph, &restored, id);
        for (index, value) in expected.iter().enumerate() {
            assert!((value - actual[index]).abs() < 1e-6, "logit {} drifted after reload", index);
        }
    }
}

/// A reloaded checkpoint must be rejected - not silently half-applied - when it
/// does not describe the live model.
#[test]
fn mismatched_checkpoints_are_rejected() {
    let mut graph = Graph::new(1);
    let mut rng = Rng::new(3);
    let _ = Lm::new(&mut graph, &mut rng, tiny_config(40));
    graph.seal_params();
    let path = unique_path("mismatch");
    ckpt::save(&path, &graph, &[]).expect("checkpoint save");

    let mut wider = Graph::new(1);
    let mut other = Rng::new(3);
    let _ = Lm::new(&mut wider, &mut other, tiny_config(41));
    wider.seal_params();
    let before: Vec<f32> = wider.val[wider.params[0].id].clone();
    let loaded = ckpt::load(&path).expect("checkpoint load");
    assert!(ckpt::apply_exact(&mut wider, &loaded).is_err(), "vocabulary change must be rejected");
    assert_eq!(wider.val[wider.params[0].id], before, "a rejected checkpoint must not mutate the model");
    let _ = std::fs::remove_file(&path);
}

/// Arbitrary bytes must survive `encode_bytes` -> `decode_bytes` exactly, and a
/// saved tokenizer must reload to the same segmentation.
#[test]
fn tokenizer_round_trips_bytes_and_files() {
    let corpus = synthetic_corpus(8_000, 5);
    let tokenizer = Bpe::train(&corpus, 300, false);
    let mut rng = Rng::new(77);
    let mut payload = vec![0u8; 4096];
    for byte in &mut payload {
        *byte = rng.next_u32() as u8;
    }
    let ids = tokenizer.encode_bytes(&payload);
    assert_eq!(tokenizer.decode_bytes(&ids).as_deref(), Some(payload.as_slice()));

    let path = unique_path("tok");
    tokenizer.save(&path).expect("tokenizer save");
    let reloaded = Bpe::load(&path).expect("tokenizer load");
    let _ = std::fs::remove_file(&path);
    assert_eq!(reloaded.vocab_size(), tokenizer.vocab_size());
    assert_eq!(reloaded.encode(&corpus[..2000]), tokenizer.encode(&corpus[..2000]));
}

/// Lion is a supported optimizer, not decoration: it has to reduce a loss too.
#[test]
fn lion_optimizer_descends() {
    let mut graph = Graph::new(1);
    let weights = graph.param("w", vec![8], vec![0.0f32; 8], true);
    graph.seal_params();
    let target: Vec<f32> = (0..8).map(|i| (i as f32 - 3.5) * 0.25).collect();
    let mut optimizer = Lion::new(&graph, 0.0);
    let mut last = f32::INFINITY;
    for _ in 0..600 {
        graph.reset();
        graph.zero_grad();
        let loss = graph.mse(weights, &target);
        graph.backward(loss);
        optimizer.step(&mut graph, 3e-3);
        last = graph.scalar(loss);
    }
    assert!(last < 1e-3, "Lion failed to converge: {}", last);
}

/// Greedy decoding is deterministic and nucleus filtering never returns a token
/// outside the vocabulary.
#[test]
fn sampling_controls_stay_in_range() {
    let mut rng = Rng::new(21);
    let vocab = 64usize;
    let base: Vec<f32> = (0..vocab).map(|i| ((i % 7) as f32) - 3.0).collect();

    let greedy = SampleCfg { temperature: 1.0, top_k: 0, top_p: 1.0, rep_penalty: 1.0, rep_window: 0, greedy: true };
    let mut logits = base.clone();
    let first = sample_token(&mut logits, &greedy, &[], &mut rng);
    let mut logits = base.clone();
    let second = sample_token(&mut logits, &greedy, &[], &mut rng);
    assert_eq!(first, second, "greedy decoding must be deterministic");

    let stochastic = SampleCfg { temperature: 0.8, top_k: 5, top_p: 0.9, rep_penalty: 1.2, rep_window: 8, greedy: false };
    let history: Vec<u32> = vec![1, 1, 2, 3];
    for _ in 0..200 {
        let mut logits = base.clone();
        let token = sample_token(&mut logits, &stochastic, &history, &mut rng);
        assert!((token as usize) < vocab, "sampled token {} outside vocabulary", token);
    }
}

/// `Decoder::reset` has to return the recurrent state to exactly the state a
/// brand-new decoder would have; otherwise reusing one decoder across prompts
/// silently leaks context from the previous prompt.
#[test]
fn decoder_reset_matches_a_fresh_decoder() {
    let cfg = tiny_config(64);
    let mut rng = Rng::new(21);
    let mut graph = Graph::new(1);
    let model = Lm::new(&mut graph, &mut rng, cfg);
    graph.seal_params();

    let prompt: Vec<u32> = vec![3, 17, 42, 5, 5, 9];
    let mut decoder = Decoder::new(&cfg);
    let mut baseline = Vec::new();
    for &id in &prompt {
        baseline.push(decoder.step(&graph, &model, id).to_vec());
    }

    // Feed unrelated tokens, then reset and replay the original prompt.
    for id in [61u32, 2, 8, 8, 8] {
        let _ = decoder.step(&graph, &model, id);
    }
    decoder.reset();
    for (index, &id) in prompt.iter().enumerate() {
        let row = decoder.step(&graph, &model, id);
        for (token, &value) in row.iter().enumerate() {
            assert!(
                (value - baseline[index][token]).abs() < 1e-6,
                "reset decoder diverged at step {} token {}: {} vs {}",
                index,
                token,
                value,
                baseline[index][token]
            );
        }
    }
}
