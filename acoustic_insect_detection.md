# Acoustic Insect Detection — Research Summary

**Prepared:** March 2026

---

## 1. Signal Types Used

Insect-generated sounds fall into three broad categories:

**Active acoustic signals (intentional communication)**

- **Stridulation** — rubbing body parts together; the dominant mechanism in Orthoptera (crickets, grasshoppers, katydids). Roughly 16,000 species use stridulatory signals, often spanning into ultrasound up to 150 kHz.
- **Tymbal mechanisms** — found in cicadas; membranes buckle rapidly to produce loud pulses.
- **Substrate-borne vibrations** — some insects communicate through plant stems or soil rather than air.

**Incidental flight sounds (by-products of locomotion)**

- **Wingbeat frequency (WBF)** — the most widely exploited signal for flying insects. Values vary widely by species: honey bees ~230–250 Hz, mosquitoes ~300–800 Hz (species and sex dependent), some beetles 50–100 Hz, flies 100–200 Hz.
- **Harmonics** — WBF generates overtones at integer multiples; systems routinely resolve up to the 5th harmonic for discrimination.

**Substrate / structural sounds**

- **Larval feeding and movement** — wood-boring larvae produce scraping and chewing sounds inside plant stems, grain kernels, and wood. Amplitudes as low as 23 dB SPL (weevil larvae in grain).
- **Movement in granular media** — grain pests crawl and feed, generating low-amplitude but detectable vibrations.

---

## 2. Key Studies and Research Groups

**Systematic reviews**

- Kohlberg, G. et al. (2024). "From buzzes to bytes: A systematic review of automated bioacoustics models used to detect, classify and monitor insects." *Journal of Applied Ecology*. Analysed 176 studies covering 302 insect species across 9 orders.

**Mosquito-focused work — HumBug / University of Oxford**

- Kiskin, I. et al. (2021). "HumBugDB: A Large-scale Acoustic Mosquito Dataset." arXiv:2110.07607. 20 hours of labelled audio, 36 species, recordings from Tanzania, Thailand, Kenya, USA and UK. Bayesian CNNs applied.
- MosquitoSong+ (PLOS ONE, 2024) — noise-robust deep learning for mosquito species/sex classification from raw wingbeat audio.
- Deep learning pipeline for mosquito detection (*Multimedia Tools and Applications*, Springer, 2022/2023) — 83% accuracy at 8 kHz vs. 90% at 96 kHz.
- CNN-based real-time mosquito genus identification (ScienceDirect, 2024) — binary Aedes/Culex: 91.76%; multiclass sex+species: 87.16%.
- Lightweight 1D CNN + LSTM for IoT mosquito classification (ACM, 2021) — 5 species, >93% accuracy.

**Bee and hive acoustics**

- "Towards acoustic monitoring of bees: wingbeat sounds are related to species and individual traits." *Philosophical Transactions of the Royal Society B*, 2024.
- "Automated Beehive Acoustics Monitoring: A Comprehensive Review." *Applied Sciences*, MDPI, 2022.
- "A deep learning-based approach for bee sound identification." ScienceDirect, 2023.

**Stored product insects — Mankin / USDA-ARS**

- Mankin, R.W. et al. (2021). "Automated Applications of Acoustics for Stored Product Insect Detection, Monitoring, and Management." *Insects* 12(3):259.
- Mankin, R.W. et al. (2011). "Perspective and Promise: A Century of Insect Acoustic Detection and Monitoring." USDA ARS.
- "Application of a multi-layer CNN to classify major insect pests in stored rice." *Computers and Electronics in Agriculture*, 2024. 84.51% accuracy for rice weevil, lesser grain borer and red flour beetle.

**Optical / lidar work — Brydegaard / Lund University**

- Multiple papers on entomological Scheimpflug lidar (dual-band 808/980 nm). Deployed in tropical cloud forests; detects insect activity up to 200 m range.
- "Insect diversity estimation in polarimetric lidar." *PLOS ONE*, 2024.
- "Automating insect monitoring using unsupervised near-infrared sensors." *Scientific Reports*, 2022.

**Multimodal — KInsecta**

- Tschaikner, M., Brandt, D. et al. (2023/2024). "Multisensor Data Fusion for Automatized Insect Monitoring (KInsecta)." arXiv:2404.18504. Camera + optical wingbeat + environmental sensors on Raspberry Pi 4.

**Open datasets**

- InsectSound1000 (PMC, 2024) — large open dataset for deep learning acoustic insect recognition.
- InsectSet459 (arXiv, 2025) — open bioacoustic ML benchmarking dataset.

---

## 3. ML/AI Approaches Applied

**Dominant paradigm — CNN on spectrograms**

Bioacoustic audio is converted to spectrograms (STFT, mel-spectrogram, log mel-spectrogram) and treated as an image classification problem. CNNs dominate the literature; the ResNet family is most common. Transfer learning from pretrained models (e.g., BirdNET based on ResNet) is increasingly used where labelled insect data is scarce.

**Specific architectures**

| Architecture | Notes |
|---|---|
| 1D CNN on raw waveform | Avoids preprocessing; suited for IoT |
| 1D CNN + LSTM | Captures temporal patterns |
| ResNet / EfficientNetV2 | State-of-the-art on InsectSet459 |
| PaSST Transformer | Audio spectrogram Transformer; competitive with CNNs |
| Bayesian CNN | Used in HumBug for uncertainty quantification |
| Hopfield network (2025) | Novel lightweight bioacoustic detection approach |

**Feature extraction methods**

MFCC, STFT spectrograms, mel-spectrograms, log mel-spectrograms, wavelet transforms, FFT for fundamental frequency and harmonic extraction.

---

## 4. Hardware Used

**Microphones**

- Standard condenser and electret microphones (broadband, 20 Hz – 20 kHz)
- MEMS microphones — low cost, suitable for IoT and embedded systems
- High-sensitivity low-noise microphones (required for quiet insects such as weevil larvae at 23 dB SPL)

**Piezoelectric contact sensors**

- PZT (lead zirconate titanate) probe sensors — inserted into grain mass for stored product monitoring
- PVDF piezoelectric film transducers
- Accelerometers — used for beehive vibration monitoring
- AED 2010L (Acoustic Emission Consultant, Fair Oaks, CA) — standard piezo probe for grain, well-validated in literature

**Optical / pseudo-acoustic sensors**

- Infrared LED sensors (dual-wavelength, near-IR at kHz sampling) — intercept insects crossing a beam; WBF extracted from modulated backscatter
- Entomological Scheimpflug lidar (Lund University) — dual-band 808/980 nm; resolves WBF, wing morphology, polarisation and melanization up to hundreds of metres
- Passive kHz lidar — measures scattered sunlight

**Platforms**

- HumBug's MozzWear app — mosquito WBF detection on budget Android smartphones
- Raspberry Pi 4 — used in the open-source KInsecta system

---

## 5. Insects Studied

| Group | Species / Notes |
|---|---|
| **Mosquitoes** | *Aedes aegypti*, *Ae. albopictus*, *Ae. gambiae*, *Culex quinquefasciatus*, *Anopheles* spp. WBF 300–800 Hz; strongly sex-dimorphic. HumBugDB covers 36 species. |
| **Bees** | *Apis mellifera* (~230–250 Hz), bumblebees, wild bees. Focus: queen presence, swarming, hive health, pesticide exposure. |
| **Stored grain beetles** | Rice weevil, khapra beetle, lesser grain borer, red flour beetle, drugstore beetle. |
| **Wood borers** | Various trunk-boring larvae (forest pests); bromeliads stem insects (USDA). |
| **Orthoptera** | Crickets, grasshoppers, katydids — best-studied for bioacoustics; ~16,000 acoustically communicating species. |
| **Cicadas** | Tymbal producers; deep learning classification 77–99% accuracy. |
| **Agricultural pests** | Aphids and hemipterans (substrate vibration); flies in oilseed rape (optical sensor, >80%). |

---

## 6. Reported Accuracy

| Study / System | Task | Accuracy |
|---|---|---|
| Kohlberg et al. review (2024) | General best-of-class ML models | >90% across hundreds of species |
| Deep learning mosquito species (Springer, 2022) | 6 mosquito species | up to 97% |
| 1D CNN + LSTM mosquito (ACM, 2021) | 5 species / sex | >93% |
| CNN real-time mosquito genus (2024) | Binary Aedes / Culex | 91.8% |
| CNN multiclass mosquito sex+species (2024) | 4 classes | 87.2% |
| Bee queen detection CNN (2021) | Queen presence | 96% |
| Beehive IoT edge ML | Colony health | 98.8% / F1=0.98 |
| Multi-layer CNN stored rice pests (2024) | 3 pest species | 84.5% |
| AED 2010L piezo probe (grain, commercial) | 1–2 insects/kg grain | 72–100% |
| Optical sensor in oilseed rape field | Insect flight classification | >80% |
| Cicada deep learning | Species classification | 77–99% |

---

## 7. Commercial and Open-Source Systems

**Commercial**

- **OSBeehives BuzzBox** — consumer IoT beehive monitor; acoustic + temperature/humidity; smartphone app with AI health assessment.
- **AED 2010L** (Acoustic Emission Consultant) — commercial piezoelectric probe for grain storage; well-validated in peer-reviewed literature.
- **Pest Probe Detector** (Sound Technology) — commercial waveguide probe for grain stores.
- **Smart mosquito trap** (US Patent US20220104474A1) — PIR + microphone + camera integrated at trap entrance.

**Academic / open-source**

- **HumBug** (University of Oxford) — MozzWear smartphone app; HumBugDB dataset on Zenodo; code on GitHub. <https://humbug.ox.ac.uk>
- **KInsecta** — open-source multi-sensor insect monitor (camera + optical wingbeat sensor + environmental sensors on Raspberry Pi 4); instructions on GitLab. arXiv:2404.18504
- **InsectSound1000** — open dataset for acoustic ML research. PMC, 2024.
- **InsectSet459** — open bioacoustic benchmark dataset. arXiv, 2025.

---

## 8. Limitations and Challenges

**Signal quality**

- Insect sounds are extremely low amplitude; weevil larvae in grain at ~23 dB SPL fall below typical office noise.
- Wind, rain, other animals, machinery and human speech all mask insect signals.
- Multiple overlapping species produce unresolvable mixtures in uncontrolled field settings.
- Microphones are omnidirectional in field use.

**Environmental confounds**

- WBF changes with temperature, humidity, age, feeding status and body condition of the individual insect.
- MosquitoSong+ (2024) explicitly notes that temperature and humidity were uncontrolled in most existing datasets.
- Geographic and population-level variation in acoustic signatures can undermine models trained in one region and deployed in another.

**Dataset scarcity**

- The vast majority of bioacoustic research has focused on vertebrates; insect datasets are sparse and often heavily imbalanced across species.
- Manual annotation is expensive; the HumBug Zooniverse project used crowdsourcing (>1,000 contributors) to address this.

**Deployment constraints**

- Deep CNN models are too resource-intensive for low-power embedded IoT sensors.
- Reducing sample rate from 96 kHz to 8 kHz drops traditional ML accuracy from ~90% to ~64% (deep learning is more robust: ~83%).
- Models validated in laboratory conditions typically perform worse in real field deployments.

---

## 9. Multimodal — Acoustic Combined with Visual / Optical Detection

**Optical wingbeat sensing (pseudo-acoustic)**

Several groups use infrared LED beams rather than microphones to extract WBF. The insect interrupts or backscatters the beam, and the modulation pattern encodes WBF and harmonics. This approach is far more noise-resistant than microphones in field conditions.

**Lund University Scheimpflug lidar**

Scheimpflug lidar at 808/980 nm simultaneously extracts WBF, harmonics, wing-body ratio, melanization index, polarimetric properties and spectral reflectance from individual insects in free flight at ranges up to 200 m. This is arguably the most information-rich single-sensor system yet demonstrated.

**KInsecta (camera + optical wingbeat, 2024)**

Combines a 10-micrometre-resolution camera (image-based species classification) with an optoacoustic wingbeat sensor, synchronised by timestamp. Fusion improves accuracy over either modality alone. Open-source, Raspberry Pi-based. arXiv:2404.18504

**Smart mosquito traps (patented)**

PIR sensor triggers both a camera and a microphone simultaneously when an insect enters. FFT is applied to both the optical and acoustic signals; agreement of the resulting frequency estimates cross-validates detection. US Patent US20220104474A1.

**Optical sensor for mosquito classification (Parasites & Vectors, 2022)**

A prototype optical sensor coupled to a commercial suction trap recorded 4,300+ *Aedes* and *Culex* mosquitoes in laboratory conditions, classifying by genus and sex with high accuracy.

**Optical sensor + ML for precision agriculture (Scientific Reports, 2021)**

Approximately 10,000 flight records from insects in oilseed rape using an optical remote sensor; tested three classification methods; achieved >80% accuracy.

---

## 10. Relevance to Camera-Based Insect Trap Systems

The literature suggests a natural upgrade path for camera-based traps is adding an **optical wingbeat sensor** — a simple IR LED and photodiode pair mounted across the trap opening — rather than a microphone. Key advantages:

- **Noise robustness** — immune to wind, rain, and background acoustic noise that defeat microphones in field conditions.
- **Direct WBF extraction** — the insect modulates the beam; WBF is extracted directly from the signal envelope, requiring minimal DSP.
- **Low cost and low power** — an IR LED + photodiode costs pennies and uses milliwatts, easily added to an existing Pi-based system.
- **Established precedent** — the KInsecta project demonstrates this approach working on a Raspberry Pi alongside a camera, closely matching existing trap hardware.
- **Complementary information** — wingbeat frequency is largely independent of image features (colour, shape), so the two modalities are complementary for species discrimination.

The most directly applicable reference is the KInsecta project (arXiv:2404.18504, 2024), which is open-source and uses commodity hardware comparable to a Raspberry Pi 5 + camera trap setup.

---

## Key References

1. Kohlberg et al. (2024). "From buzzes to bytes." *Journal of Applied Ecology*. <https://besjournals.onlinelibrary.wiley.com/doi/10.1111/1365-2664.14630>
2. Kiskin et al. (2021). "HumBugDB." arXiv:2110.07607. <https://arxiv.org/abs/2110.07607>
3. HumBug Project — University of Oxford. <https://humbug.ox.ac.uk>
4. MosquitoSong+. *PLOS ONE*, 2024. <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121>
5. Deep learning mosquito pipeline. *Springer*, 2022. <https://link.springer.com/article/10.1007/s11042-022-13367-0>
6. CNN mosquito genus identification. ScienceDirect, 2024. <https://www.sciencedirect.com/science/article/pii/S1574954124000372>
7. 1D CNN + LSTM mosquito classification. ACM, 2021. <https://dl.acm.org/doi/10.1145/3462203.3475908>
8. Acoustic monitoring of bees. *Phil. Trans. R. Soc. B*, 2024. <https://royalsocietypublishing.org/doi/10.1098/rstb.2023.0111>
9. Mankin et al. (2021). "Automated acoustics for stored product insects." *Insects* 12(3):259. <https://www.mdpi.com/2075-4450/12/3/259>
10. Multi-layer CNN stored grain pests. *Computers and Electronics in Agriculture*, 2024. <https://www.sciencedirect.com/science/article/pii/S0168169924006884>
11. KInsecta multisensor fusion. arXiv:2404.18504, 2024. <https://arxiv.org/abs/2404.18504>
12. Scheimpflug lidar — Brydegaard group. PMC, 2023. <https://pmc.ncbi.nlm.nih.gov/articles/PMC10274244/>
13. Insect diversity in polarimetric lidar. *PLOS ONE*, 2024. <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0312770>
14. Automating insect monitoring with near-infrared sensors. *Scientific Reports*, 2022. <https://www.nature.com/articles/s41598-022-06439-6>
15. InsectSound1000 dataset. PMC, 2024. <https://pmc.ncbi.nlm.nih.gov/articles/PMC11082239/>
16. InsectSet459 dataset. arXiv, 2025. <https://arxiv.org/pdf/2503.15074>
17. Optical sensor for mosquito classification. *Parasites & Vectors*, 2022. <https://parasitesandvectors.biomedcentral.com/articles/10.1186/s13071-022-05324-5>
18. Flying insect detection with optical sensors. *Scientific Reports*, 2021. <https://www.nature.com/articles/s41598-021-81005-0>
19. Smart mosquito trap patent. US20220104474A1. <https://patents.google.com/patent/US20220104474A1/en>
