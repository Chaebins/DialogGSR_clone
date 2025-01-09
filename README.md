<p align="center">
  <h1 align="center">DialogGSR: Generative Subgraph Retrieval for Knowledge Graph-Grounded Dialog Generation</h1>
  
  <p align="center">Jinyoung Park, Minseok Joo, Joo-Kyung Kim, Hyunwoo J. Kim.
  </p>

  <h3 align="center">
    <a href="https://arxiv.org/pdf/2410.09350" target='_blank'><img src="https://img.shields.io/badge/arXiv-2410.09350-b31b1b.svg"></a>
  </h3>

</p>
Official PyTorch implementation of the "DialogGSR: Generative Subgraph Retrieval for Knowledge Graph-Grounded Dialog Generation".
(EMNLP 2024)


## Enviroment
To install requirements, run:
```bash
git clone https://github.com/mlvlab/DialogGSR.git
cd DialogGSR
conda create -n dialoggsr python==3.9
conda activate dialoggsr
sh setting.sh
```

---

## Preparation
### OpenDialKG
We utilized OpenDialKG dataset from the [link](https://github.com/facebookresearch/opendialkg) repository. Place the downloaded dataset in the `data/` folder.

---

## Training

```bash
python main.py --output_dir <output_dir>
```

## Contact
If you have any questions, please create an issue on this repository or contact at lpmn678@korea.ac.kr.

## Citation
If you find our work interesting, please consider giving a ⭐ and citation.
```bibtex
@inproceedings{park2024generative,
  title={Generative Subgraph Retrieval for Knowledge Graph-Grounded Dialog Generation},
  author={Park, Jinyoung and Joo, Minseok and Kim, Joo-Kyung and Kim, Hyunwoo J},
  booktitle={EMNLP},
  year={2024}
}
```
