import argparse
import os
import re
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('logs',   exist_ok=True)
os.makedirs('plots',  exist_ok=True)
os.makedirs('saved',  exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('data/speech',          exist_ok=True)
os.makedirs('data/speechcommands',  exist_ok=True)
os.makedirs('data/voice_clone',     exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/run.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

#Part 1: Speech Tokenizer

class SpeechTokenizer:
    """
    Character-level tokenizer for TTS.
    Handles: text normalization, special tokens, accent tags.
    """
    ACCENTS = ['[EN-US]', '[EN-BR]', '[EN-INDIA]', '[EN-AU]', '[EN-DEFAULT]']

    def __init__(self):
        chars = " !',-.?abcdefghijklmnopqrstuvwxyz"
        self.vocab = {c: i+3 for i, c in enumerate(sorted(chars))}
        self.vocab['<PAD>'] = 0
        self.vocab['<BOS>'] = 1
        self.vocab['<EOS>'] = 2
        for i, a in enumerate(self.ACCENTS):
            self.vocab[a] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def normalize(self, text):
        text = text.lower()
        text = re.sub(r'dr\.', 'doctor', text)
        text = re.sub(r'mr\.', 'mister', text)
        text = re.sub(r'(\d+)', lambda m: self._num_to_words(int(m.group())), text)
        text = re.sub(r"[^a-z !',\-.?\[\]]", '', text)
        return text.strip()

    def _num_to_words(self, n):
        words = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',
                 6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}
        return words.get(n, str(n))

    def encode(self, text, add_special=True):
        tag_pattern = '|'.join(re.escape(a) for a in self.ACCENTS)
        parts = re.split(f'({tag_pattern})', text)
        tokens = []
        if add_special:
            tokens.append(self.vocab['<BOS>'])
        for part in parts:
            if part in self.ACCENTS:
                tokens.append(self.vocab[part])
            else:
                for ch in self.normalize(part):
                    if ch in self.vocab:
                        tokens.append(self.vocab[ch])
        if add_special:
            tokens.append(self.vocab['<EOS>'])
        return tokens

    def decode(self, ids):
        return ''.join(self.inv_vocab.get(i, '?') for i in ids
                       if i not in (self.vocab['<PAD>'], self.vocab['<BOS>'], self.vocab['<EOS>']))

    def __len__(self):
        return len(self.vocab)


def run_tokenizer():
    """Exercise 1: tokenize sentences and print results table."""
    tokenizer = SpeechTokenizer()
    log.info(f"Vocabulary size: {len(tokenizer)}")

    sentences = [
        "Hello, how are you?",
        "Dr. Smith prescribed 10 tablets.",
        "[EN-US] I got the job!",
        "[EN-BR] I lost my wallet.",
        "[EN-INDIA] This is completely unacceptable!",
    ]

    log.info(f"\n{'Sentence':<45} {'#Char':>6} {'#Total':>7} {'Accent Tag'}")
    log.info("-" * 75)
    for sent in sentences:
        ids      = tokenizer.encode(sent)
        ids_bare = tokenizer.encode(sent, add_special=False)
        tag      = next((a for a in SpeechTokenizer.ACCENTS if a in sent), '—')
        tag_id   = tokenizer.vocab.get(tag, '—')
        # char count = tokens without BOS/EOS and without the accent tag token
        n_char   = len([i for i in ids_bare if i != tokenizer.vocab.get(tag)])
        log.info(f"{sent[:43]:<45} {n_char:>6} {len(ids):>7}   {tag} (id={tag_id})")

    log.info("\n--- Detail: '[EN-US] I got the job!' ---")
    sent   = "[EN-US] I got the job!"
    ids    = tokenizer.encode(sent)
    log.info(f"IDs:     {ids}")
    log.info(f"Decoded: {tokenizer.decode(ids)}")
    log.info(f"[EN-US] = single token ID {tokenizer.vocab['[EN-US]']}")

#Part 2: Mel Spectrogram 

def run_melspectrogram():
    import urllib.request
    log.info("=== Part 2: Mel Spectrogram ===")

    wav_path = 'data/speech/sample_speech.wav'
    url      = 'https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'
    if not os.path.exists(wav_path):
        log.info("  Downloading sample WAV...")
        try:
            urllib.request.urlretrieve(url, wav_path)
        except Exception as e:
            log.warning(f"  Download failed: {e}")
            log.warning("  Place a WAV at data/speech/sample_speech.wav and re-run.")
            return

    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        waveform    = T.Resample(sample_rate, 16000)(waveform)
        sample_rate = 16000
    log.info(f"  Loaded waveform: {waveform.shape}, sample_rate={sample_rate}")

    mel_tf   = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80)
    mel_spec = mel_tf(waveform[0].unsqueeze(0)).squeeze()
    log_mel  = torch.log(mel_spec + 1e-9)
    log.info(f"  Mel spectrogram shape: {log_mel.shape}  (80 bins x {log_mel.shape[1]} frames)")

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    axes[0].plot(waveform[0].numpy(), linewidth=0.4, color='steelblue')
    axes[0].set_title('Raw Waveform (16,000 samples/sec)')
    axes[0].set_xlabel('Sample index')

    im = axes[1].imshow(log_mel.numpy(), aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title('Log Mel Spectrogram (80 bins × time frames)')
    axes[1].set_xlabel('Time frames (16ms each)')
    axes[1].set_ylabel('Mel frequency bins')
    plt.colorbar(im, ax=axes[1], label='Log Energy')
    plt.tight_layout()
    plt.savefig('plots/mel_spectrogram.png', dpi=150)
    plt.close()
    log.info("  Plot saved to plots/mel_spectrogram.png")

# ── Part 3: CTC ─────────────────────────────────────────────────────────────

BLANK = '_'

def ctc_collapse(alignment):
    """Merge consecutive duplicates, then remove blanks."""
    merged = []
    for ch in alignment:
        if not merged or ch != merged[-1]:
            merged.append(ch)
    return ''.join(ch for ch in merged if ch != BLANK)


NEG_INF = -1e9

def log_add(a, b):
    if a == NEG_INF: return b
    if b == NEG_INF: return a
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def ctc_forward_log_prob(log_probs, labels, blank=0):
    """
    log_probs: (T, V) log-softmax output per frame
    labels:    list of label indices (no blanks)
    Returns:   log P_CTC(labels | log_probs)
    """
    T, V = log_probs.shape
    ext  = [blank]
    for lab in labels:
        ext += [lab, blank]
    S = len(ext)

    alpha = np.full((T, S), NEG_INF)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            stay = alpha[t-1, s]
            prev = alpha[t-1, s-1] if s - 1 >= 0 else NEG_INF
            skip = NEG_INF
            if s - 2 >= 0 and ext[s] != blank and ext[s] != ext[s-2]:
                skip = alpha[t-1, s-2]
            best_prev   = log_add(log_add(stay, prev), skip)
            alpha[t, s] = best_prev + log_probs[t, ext[s]]

    if S == 1:
        return alpha[T-1, S-1]
    return log_add(alpha[T-1, S-1], alpha[T-1, S-2])

#Toy CTC dataset

ALPHABET   = list('helo wrd')
CHAR2IDX   = {c: i+1 for i, c in enumerate(ALPHABET)}
IDX2CHAR   = {i+1: c for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 1
N_MELS     = 20
WORDS      = ['hello', 'world', 'hero', 'red', 'led', 'doer']


def synthesize_frames(word, frames_per_char=(3, 8)):
    frames, char_at_frame = [], []
    for ch in word:
        n    = random.randint(*frames_per_char)
        base = np.zeros(N_MELS)
        base[CHAR2IDX[ch] % N_MELS] = 3.0
        for _ in range(n):
            frames.append(base + np.random.randn(N_MELS) * 0.5)
            char_at_frame.append(ch)
    return np.stack(frames), char_at_frame


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp  = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev  = temp
    return dp[n]


class TinyCTCModel(nn.Module):
    def __init__(self, in_dim=N_MELS, hidden=64, vocab=VOCAB_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(hidden * 2, vocab)

    def forward(self, x):
        h, _ = self.lstm(x)
        return F.log_softmax(self.fc(h), dim=-1)


def run_ctc(epochs=300):
    log.info("=== Part 3: CTC ===")

    # --- Exercise 2a: collapsing function ---
    log.info("\n-- Collapsing function --")
    examples = [
        list('HHEELLLLOO'),
        list('H_EE_LL_LO'),
        list('H_E_L_L_O'),
        list('HHHHEEEELLLLLLOOOO'),
    ]
    for align in examples:
        log.info(f"  {''.join(align):22} -> {ctc_collapse(align)}")

    # --- Exercise 2b: forward algorithm on two words ---
    log.info("\n-- Forward algorithm (Exercise 2b) --")
    np.random.seed(0)
    logits    = np.random.randn(6, 5)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    for word, label_ids in [('HEL', [1,2,3]), ('LEH', [3,2,1])]:
        lp = ctc_forward_log_prob(log_probs, label_ids)
        log.info(f"  log P_CTC('{word}') = {lp:.4f}   P = {np.exp(lp):.6f}")

    # --- Train tiny CTC model ---
    log.info(f"\n-- Training TinyCTCModel for {epochs} steps --")
    model       = TinyCTCModel()
    optimizer   = torch.optim.Adam(model.parameters(), lr=1e-2)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    losses, cers = [], []
    for step in range(epochs):
        word      = random.choice(WORDS)
        frames, _ = synthesize_frames(word)
        x         = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
        targets   = torch.tensor([CHAR2IDX[c] for c in word], dtype=torch.long)

        log_p          = model(x).transpose(0, 1)
        input_lengths  = torch.tensor([log_p.size(0)])
        target_lengths = torch.tensor([len(targets)])

        loss = ctc_loss_fn(log_p, targets, input_lengths, target_lengths)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            pred_ids   = model(x).squeeze(0).argmax(dim=-1).tolist()
            pred_chars = [IDX2CHAR.get(i, BLANK) if i != 0 else BLANK for i in pred_ids]
            decoded    = ctc_collapse(pred_chars)
        cer = edit_distance(decoded, word) / max(len(word), 1)
        cers.append(cer)
        model.train()

        if (step + 1) % 50 == 0:
            log.info(f"  Step {step+1:3d} | loss: {np.mean(losses[-50:]):.4f} | CER: {np.mean(cers[-50:])*100:.1f}%")

    below10 = next((i+1 for i, c in enumerate(cers) if c < 0.1), None)
    log.info(f"\n  CER first dropped below 10% at step: {below10}")

    torch.save(model.state_dict(), 'saved/ctc_model.pth')
    log.info("  Model saved to saved/ctc_model.pth")

    # --- Plots ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(losses)
    axes[0].set_title('CTC Training Loss'); axes[0].set_ylabel('Loss'); axes[0].grid(True)
    axes[1].plot([c*100 for c in cers])
    axes[1].axhline(10, color='r', linestyle='--', label='10% CER')
    axes[1].set_title('Character Error Rate'); axes[1].set_ylabel('CER (%)'); axes[1].set_xlabel('Step')
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('plots/ctc_training.png', dpi=150)
    plt.close()
    log.info("  Plot saved to plots/ctc_training.png")

    # --- Greedy decoding grid ---
    model.eval()
    fig, axes = plt.subplots(len(WORDS), 1, figsize=(12, 2 * len(WORDS)))
    for ax, word in zip(axes, WORDS):
        frames, _ = synthesize_frames(word)
        x         = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            log_p = model(x).squeeze(0)
        pred_ids   = log_p.argmax(dim=-1).tolist()
        pred_chars = [IDX2CHAR.get(i, BLANK) if i != 0 else BLANK for i in pred_ids]
        decoded    = ctc_collapse(pred_chars)
        colors     = plt.cm.tab10(np.linspace(0, 1, len(ALPHABET) + 1))
        for t, ch in enumerate(pred_chars):
            idx = 0 if ch == BLANK else ALPHABET.index(ch) + 1
            ax.bar(t, 1, color=colors[idx], edgecolor='white', linewidth=0.3)
            if ch != BLANK:
                ax.text(t, 0.5, ch, ha='center', va='center', fontsize=8, color='white')
        ax.set_xlim(0, len(pred_chars)); ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xticks([])
        status = 'correct' if decoded == word else 'wrong'
        ax.set_ylabel(f'"{word}"', rotation=0, labelpad=35, fontsize=10, va='center')
        ax.set_title(f'Raw ({len(pred_chars)} frames) -> "{decoded}"  [{status}]', fontsize=9, loc='left')
    plt.suptitle('CTC Greedy Decoding', fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig('plots/ctc_decoding.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  Decoding grid saved to plots/ctc_decoding.png")

#Part 4: wav2vec2 Linear Probe

def run_wav2vec2(classes='yes,no,stop,go'):
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    from sklearn.model_selection import train_test_split
    from sklearn.manifold import TSNE

    log.info("=== Part 4: wav2vec2 Linear Probe ===")

    PROBE_WORDS = classes.split(',')
    N_PER_CLASS = 40
    W2V_NAME    = os.path.expanduser('~/.cache/huggingface/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8')

    log.info(f"  Classes: {PROBE_WORDS}  ({N_PER_CLASS} clips/class)")
    log.info(f"  Loading pretrained {W2V_NAME}...")
    w2v_extractor = Wav2Vec2FeatureExtractor.from_pretrained(W2V_NAME, local_files_only=True)
    w2v_model     = Wav2Vec2Model.from_pretrained(W2V_NAME, local_files_only=True).to(device).eval()
    for p in w2v_model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in w2v_model.parameters())
    log.info(f"  Loaded {n_params:,} frozen parameters")

    log.info("  Loading SpeechCommands dataset...")
    sc_dataset = torchaudio.datasets.SPEECHCOMMANDS(root='data/speechcommands', download=True)

    by_label = {w: [] for w in PROBE_WORDS}
    for i in range(len(sc_dataset)):
        wvf, sr, label, *_ = sc_dataset[i]
        if label in by_label and len(by_label[label]) < N_PER_CLASS:
            by_label[label].append(wvf)
        if all(len(v) >= N_PER_CLASS for v in by_label.values()):
            break

    log.info("  Extracting frozen wav2vec2 features...")
    feats, labels_list = [], []
    with torch.no_grad():
        for label, clips in by_label.items():
            for wvf in clips:
                inputs = w2v_extractor(
                    wvf.squeeze(0).numpy(), sampling_rate=16000, return_tensors='pt'
                ).to(device)
                out    = w2v_model(**inputs).last_hidden_state
                pooled = out.mean(dim=1).squeeze(0).cpu()
                feats.append(pooled)
                labels_list.append(PROBE_WORDS.index(label))

    X = torch.stack(feats)
    y = torch.tensor(labels_list)
    log.info(f"  Features: {X.shape[0]} clips x {X.shape[1]} dims")

    log.info("  Extracting raw mel-spectrogram baseline features...")
    mel_tf    = T.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)
    mel_feats, mel_labels = [], []
    for label, clips in by_label.items():
        for wvf in clips:
            mel  = mel_tf(wvf)
            pool = mel.mean(dim=-1).squeeze(0)
            mel_feats.append(pool)
            mel_labels.append(PROBE_WORDS.index(label))
    X_mel = torch.stack(mel_feats)
    y_mel = torch.tensor(mel_labels)

    def train_linear_probe(X_feat, y_feat, n_classes, tag):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_feat.numpy(), y_feat.numpy(), test_size=0.3, random_state=42, stratify=y_feat.numpy())
        X_tr = torch.tensor(X_tr, dtype=torch.float32)
        y_tr = torch.tensor(y_tr, dtype=torch.long)
        X_te = torch.tensor(X_te, dtype=torch.float32)
        y_te = torch.tensor(y_te, dtype=torch.long)

        probe = nn.Linear(X_feat.shape[1], n_classes)
        opt   = torch.optim.Adam(probe.parameters(), lr=1e-2)
        for _ in range(100):
            loss = F.cross_entropy(probe(X_tr), y_tr)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (probe(X_te).argmax(1) == y_te).float().mean().item()
        log.info(f"  [{tag}] test accuracy: {acc*100:.1f}%  (random baseline: {100/n_classes:.1f}%)")
        return acc

    acc_mel = train_linear_probe(X_mel, y_mel, len(PROBE_WORDS), 'raw mel-spectrogram')
    acc_w2v = train_linear_probe(X,     y,     len(PROBE_WORDS), 'wav2vec2 frozen')
    log.info(f"  wav2vec2 improvement over mel baseline: +{(acc_w2v - acc_mel)*100:.1f}%")

    log.info("  Running t-SNE...")
    proj   = TSNE(n_components=2, random_state=42, perplexity=15).fit_transform(X.numpy())
    colors = plt.cm.tab10(np.linspace(0, 1, len(PROBE_WORDS)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, word in enumerate(PROBE_WORDS):
        mask = y.numpy() == i
        axes[0].scatter(proj[mask, 0], proj[mask, 1], c=[colors[i]], label=word, alpha=0.7, s=30)
    axes[0].legend()
    axes[0].set_title(f'wav2vec2 Embeddings (t-SNE)\nLinear probe acc: {acc_w2v*100:.1f}%')
    axes[0].axis('off')

    axes[1].bar(['Raw Mel\nSpectrogram', 'wav2vec2\n(frozen)'],
                [acc_mel*100, acc_w2v*100], color=['steelblue', 'darkorange'])
    axes[1].axhline(100/len(PROBE_WORDS), color='red', linestyle='--', label='Random baseline')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Linear Probe: wav2vec2 vs Raw Mel Features')
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    for i, v in enumerate([acc_mel*100, acc_w2v*100]):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/wav2vec2_probe.png', dpi=150)
    plt.close()
    log.info("  Plot saved to plots/wav2vec2_probe.png")

#Part 5: Voice Cloning (OpenVoice)

def run_voice_clone(mode='extract-se', reference='data/voice_clone/my_voice.wav',
                    accent='us', text='I got the job!', language='EN'):
    import sys, subprocess
    log.info("=== Part 5: Voice Cloning (OpenVoice) ===")

    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS as MeloTTS

    from huggingface_hub import snapshot_download

    log.info("  Downloading OpenVoiceV2 checkpoint (cached after first run)...")
    ckpt_dir = snapshot_download(repo_id='myshell-ai/OpenVoiceV2')
    tone_color_converter = ToneColorConverter(
        f'{ckpt_dir}/converter/config.json', device=str(device))
    tone_color_converter.load_ckpt(f'{ckpt_dir}/converter/checkpoint.pth')
    log.info("  OpenVoiceV2 loaded.")

    se_save = 'data/voice_clone/target_se.pth'

    if mode == 'extract-se':
        if not os.path.exists(reference):
            log.error(f"  Reference file not found: {reference}")
            log.error("  Record a ~10-30s WAV/MP3 and place it there.")
            return
        log.info(f"  Extracting tone color from: {reference}")
        target_se, _ = se_extractor.get_se(
            reference, tone_color_converter,
            target_dir='data/voice_clone/processed', vad=True)
        torch.save(target_se, se_save)
        log.info(f"  Saved: {se_save}  shape={target_se.shape}")

    elif mode in ('generate', 'all'):
        if not os.path.exists(se_save):
            log.error("  No tone color embedding found. Run --extract-se first.")
            return
        target_se = torch.load(se_save, map_location=device)

        style_map = {
            'us':    ('en-us.pth',    'EN-US'),
            'br':    ('en-br.pth',    'EN-BR'),
            'india': ('en-india.pth', 'EN_INDIA'),
            'au':    ('en-au.pth',    'EN-AU'),
        }
        accents   = list(style_map.keys()) if mode == 'all' else [accent]
        base_tts  = MeloTTS(language='EN', device=str(device))
        speaker_ids = dict(base_tts.hps.data.spk2id)
        log.info(f"  Text: \"{text}\"  |  Accents: {accents}")

        for acc in accents:
            se_file, spk_key = style_map[acc]
            spk_id    = speaker_ids.get(spk_key, list(speaker_ids.values())[0])
            base_path = f'data/voice_clone/base_{acc}.wav'
            out_path  = f'data/voice_clone/cloned_{acc}.wav'
            base_tts.tts_to_file(text, spk_id, base_path, speed=1.0)
            source_se = torch.load(f'{ckpt_dir}/base_speakers/ses/{se_file}', map_location=device)
            tone_color_converter.convert(
                audio_src_path=base_path, src_se=source_se,
                tgt_se=target_se, output_path=out_path, tau=0.3)
            log.info(f"  [{acc:6}] -> {out_path}")

        mel_tf = T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80)
        fig, axes = plt.subplots(1, len(accents), figsize=(5 * len(accents), 4))
        if len(accents) == 1: axes = [axes]
        for ax, acc in zip(axes, accents):
            out_path = f'data/voice_clone/cloned_{acc}.wav'
            if not os.path.exists(out_path): continue
            wvf, sr = torchaudio.load(out_path)
            if sr != 22050: wvf = T.Resample(sr, 22050)(wvf)
            mel     = mel_tf(wvf[0].unsqueeze(0)).squeeze()
            log_mel = torch.log(mel + 1e-9)
            ax.imshow(log_mel.numpy(), aspect='auto', origin='lower', cmap='magma')
            ax.set_title(f'[{acc.upper()}]', fontweight='bold')
            ax.set_xlabel('Time frames'); ax.set_ylabel('Mel bins')
        plt.suptitle('Same Cloned Voice, Different Accents', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/voice_clone_mel_grid.png', dpi=150)
        plt.close()
        log.info("  Mel grid saved to plots/voice_clone_mel_grid.png")

    elif mode == 'cross-lingual':
        if not os.path.exists(se_save):
            log.error("  No tone color embedding found. Run --extract-se first.")
            return
        target_se = torch.load(se_save, map_location=device)
        cross_texts = {
            'EN': text,
            'ES': 'Hola, esta es una prueba de clonacion de voz entre idiomas.',
            'FR': 'Bonjour, ceci est un test de clonage vocal interlingue.',
        }
        lang_se_map = {'EN': 'en-us.pth', 'ES': 'es.pth', 'FR': 'fr.pth'}
        langs = list(cross_texts.keys()) if language.upper() == 'ALL' else [language.upper()]

        for lang in langs:
            lang_text = cross_texts.get(lang, text)
            se_file   = lang_se_map.get(lang, 'en-us.pth')
            base_path = f'data/voice_clone/base_{lang}.wav'
            out_path  = f'data/voice_clone/cloned_{lang}.wav'
            base_tts  = MeloTTS(language=lang, device=str(device))
            spk_ids   = base_tts.hps.data.spk2id
            base_tts.tts_to_file(lang_text, list(spk_ids.values())[0], base_path, speed=1.0)
            se_path = f'{ckpt_dir}/base_speakers/ses/{se_file}'
            if not os.path.exists(se_path):
                log.warning(f"  No SE file for {lang}, skipping.")
                continue
            source_se = torch.load(se_path, map_location=device)
            tone_color_converter.convert(
                audio_src_path=base_path, src_se=source_se,
                tgt_se=target_se, output_path=out_path)
            log.info(f"  [{lang}] \"{lang_text}\" -> {out_path}")

    elif mode == 'cosine-sim':
        if not os.path.exists(se_save):
            log.error("  No tone color embedding found. Run --extract-se first.")
            return
        target_se = torch.load(se_save, map_location=device)
        accents   = ['us', 'br', 'india', 'au']
        log.info(f"  {'Accent':<10} {'Cosine Sim':>12}")
        log.info("  " + "-" * 25)
        sims = {}
        for acc in accents:
            out_path = f'data/voice_clone/cloned_{acc}.wav'
            if not os.path.exists(out_path):
                log.warning(f"  {acc}: file not found, skipping.")
                continue
            clone_se, _ = se_extractor.get_se(
                out_path, tone_color_converter,
                target_dir='data/voice_clone/processed', vad=False)
            cos_sim = F.cosine_similarity(
                target_se.view(1, -1), clone_se.view(1, -1)).item()
            sims[acc] = cos_sim
            log.info(f"  {acc:<10} {cos_sim:>12.4f}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(list(sims.keys()), list(sims.values()), color='darkorange')
        ax.axhline(1.0, color='green', linestyle='--', label='Perfect match')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Tone Color Consistency Across Accents')
        plt.tight_layout()
        plt.savefig('plots/cosine_sim.png', dpi=150)
        plt.close()
        log.info("  Plot saved to plots/cosine_sim.png")

    else:
        log.error(f"  Unknown mode: {mode}. Use: extract-se | generate | all | cross-lingual | cosine-sim")

#Main / Argparse

def main():
    parser = argparse.ArgumentParser(description='A6: Speech Processing')
    parser.add_argument('--model', type=str, required=True,
                        choices=['tokenizer', 'melspectrogram', 'ctc', 'wav2vec2-probe', 'voice-clone'],
                        help='Which part to run')

    # CTC
    parser.add_argument('--epochs',  type=int, default=300, help='CTC training steps')

    # wav2vec2
    parser.add_argument('--classes', type=str, default='yes,no,stop,go',
                        help='Comma-separated SpeechCommands classes for linear probe')

    # voice clone
    parser.add_argument('--extract-se',    action='store_true', help='Extract tone color from reference clip')
    parser.add_argument('--generate',      action='store_true', help='Synthesize in cloned voice')
    parser.add_argument('--cross-lingual', action='store_true', help='Cross-lingual cloning')
    parser.add_argument('--cosine-sim',    action='store_true', help='Compute cosine sim of cloned outputs')
    parser.add_argument('--reference',     type=str, default='data/voice_clone/my_voice.wav',
                        help='Path to reference voice recording')
    parser.add_argument('--accent',        type=str, default='all',
                        choices=['us', 'br', 'india', 'au', 'all'],
                        help='Accent for voice cloning')
    parser.add_argument('--text',          type=str, default='I got the job!',
                        help='Text to synthesize')
    parser.add_argument('--language',      type=str, default='ALL',
                        help='Language for cross-lingual cloning (EN, ES, FR, ALL)')

    args = parser.parse_args()

    if args.model == 'tokenizer':
        run_tokenizer()

    elif args.model == 'melspectrogram':
        run_melspectrogram()

    elif args.model == 'ctc':
        run_ctc(epochs=args.epochs)

    elif args.model == 'wav2vec2-probe':
        run_wav2vec2(classes=args.classes)

    elif args.model == 'voice-clone':
        if args.extract_se:
            run_voice_clone(mode='extract-se', reference=args.reference)
        elif args.generate:
            mode = 'all' if args.accent == 'all' else 'generate'
            run_voice_clone(mode=mode, accent=args.accent, text=args.text)
        elif args.cross_lingual:
            run_voice_clone(mode='cross-lingual', text=args.text, language=args.language)
        elif args.cosine_sim:
            run_voice_clone(mode='cosine-sim')
        else:
            log.error("  Pass one of: --extract-se | --generate | --cross-lingual | --cosine-sim")


if __name__ == '__main__':
    main()
    