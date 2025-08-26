# Citation & References

## Citing Lumi

If you use Lumi in your research or educational work, please cite:

```bibtex
@software{lumi_mini_llm,
  title={Lumi: Mini-LLM Training Pipeline},
  author={Lumi Contributors},
  year={2024},
  url={https://github.com/your-repo/Lumi},
  note={Educational LLM training toolkit for personal GPUs}
}
```

## Dataset Citations

### Pre-training Datasets

#### OpenWebText
```bibtex
@article{gokaslan2019openwebtext,
  title={OpenWebText: An Open Source Recreation of GPT-2's WebText Dataset},
  author={Gokaslan, Aaron and Cohen, Vanya},
  year={2019},
  url={https://skylion007.github.io/OpenWebTextCorpus/}
}
```
- **License**: Public Domain / No specific license
- **HuggingFace**: `openwebtext`
- **Size**: ~8M documents, ~40GB text
- **Usage in Lumi**: Primary pre-training corpus

#### WikiText-103
```bibtex
@article{merity2016pointer,
  title={Pointer Sentinel Mixture Models},
  author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  journal={arXiv preprint arXiv:1609.07843},
  year={2016}
}
```
- **License**: Creative Commons Attribution-ShareAlike 3.0
- **HuggingFace**: `wikitext-103-raw-v1`
- **Size**: ~103M tokens from Wikipedia
- **Usage in Lumi**: Alternative pre-training dataset for smaller experiments

### Evaluation Datasets

#### WikiText-2
```bibtex
@article{merity2016pointer,
  title={Pointer Sentinel Mixture Models},
  author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  journal={arXiv preprint arXiv:1609.07843},
  year={2016}
}
```
- **License**: Creative Commons Attribution-ShareAlike 3.0
- **HuggingFace**: `wikitext-2-raw-v1`
- **Size**: ~2M tokens
- **Usage in Lumi**: Perplexity evaluation benchmark

#### BoolQ
```bibtex
@inproceedings{clark2019boolq,
  title={BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions},
  author={Clark, Christopher and Lee, Kenton and Chang, Ming-Wei and Kwiatkowski, Tom and Collins, Michael and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2019}
}
```
- **License**: Creative Commons Attribution 4.0
- **HuggingFace**: `boolq`
- **Size**: ~15.9K yes/no questions
- **Usage in Lumi**: Zero-shot reasoning evaluation

### Fine-tuning Datasets (Optional)

#### BookCorpus
⚠️ **License Status**: Unclear/Disputed
- **Original source**: No longer available
- **HuggingFace**: Various reproductions with unclear licensing
- **Recommendation**: Use alternative datasets like OpenWebText for legal clarity
- **Usage in Lumi**: Not used by default due to licensing concerns

#### Alternative SFT Datasets
For supervised fine-tuning, consider these well-licensed alternatives:

- **Alpaca**: `tatsu-lab/alpaca` (CC BY-NC 4.0)
- **Open Assistant**: `OpenAssistant/oasst1` (Apache 2.0)
- **ShareGPT**: Various derivatives with different licenses

## Framework Citations

### Core Libraries

#### PyTorch
```bibtex
@article{paszke2019pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and others},
  journal={Advances in Neural Information Processing Systems},
  year={2019}
}
```

#### Transformers
```bibtex
@article{wolf2019transformers,
  title={Transformers: State-of-the-Art Natural Language Processing},
  author={Wolf, Thomas and Debut, Lysandre and Sanh, Victor and others},
  journal={arXiv preprint arXiv:1910.03771},
  year={2019}
}
```

#### TRL (Transformer Reinforcement Learning)
```bibtex
@software{vonwerra2022trl,
  title={TRL: Transformer Reinforcement Learning},
  author={von Werra, Leandro and Belkada, Younes and others},
  year={2022},
  url={https://github.com/huggingface/trl}
}
```

### Techniques

#### LoRA (Low-Rank Adaptation)
```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and others},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

#### DPO (Direct Preference Optimization)
```bibtex
@article{rafailov2023direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and others},
  journal={arXiv preprint arXiv:2305.18290},
  year={2023}
}
```

#### FlashAttention
```bibtex
@article{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Legal Notice

### Dataset Licensing Compliance
- All datasets used in Lumi maintain their original licenses
- Users must comply with individual dataset license terms
- Commercial use may require additional licensing for some datasets

### Model Outputs
- Generated text inherits training data characteristics and biases
- No warranty on accuracy, safety, or appropriateness of outputs
- Users responsible for filtering and monitoring generated content

### Acknowledgments
We thank the creators and maintainers of all datasets and libraries that make Lumi possible. This project builds upon the work of countless researchers and engineers in the open-source ML community.