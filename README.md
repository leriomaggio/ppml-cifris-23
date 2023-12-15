# PPML: Machine Learning on Data you cannot see

Repository for the [tutorial](https://schedule.mozillafestival.org/session/3TAPD8-1) on **Privacy-Preserving Machine Learning** (`PPML`) presented at [SciPy 2023](https://www.scipy2023.scipy.org/)

Privacy guarantee is the most crucial requirement when it comes to analyse
sensitive data. In fact, sensitive data could not be shared nor moved from their
silos, let alone analysed in their raw form. As a result, data anonymization
techniques are used to generate a sanitised version of the original data. These
techniques are valuable tools to allow sensitive data to be used by Machine
Learning (ML) algorithms, but these methods alone are not enough to guarantee
complete privacy protection [6]. Moreover, multiple studies have demonstrated
that ML algorithms trained on private data suffer from a persistent vulnerability
that can unintentionally expose information about training samples [1] [3] [5].
This is particularly the case of Deep Neural Networks due to a hard-to-avoid
memoization effect in their internal parameters [1].

Differential privacy (DP) [2] is a system for publicly sharing information
about a dataset by describing the patterns of groups within the data, while withholding 
information about individuals. This technique has recently attracted in-
creasing interest from the ML community, as a method to quantify the anonymization of 
sensitive data during training [3] [2]. Moreover, DP integrates seamlessly into the whole
process, with no direct effect on its reproducibility.

In this talk, we will discuss how DP methods can be effectively used for
Privacy Preserving Machine learning. We will introduce the main theoretical
foundations of DP that are relevant for ML analyses. Afterwards, we will demon-
strate how DL models could be exploited [4] (i.e. inference attack ) to reconstruct
original training data by solely analysing models predictions, and how DP can
help to protect the privacy of our model, with minimal disruption to the original
training pipeline. Final remarks on more complex ML training and inference sce-
narios will be examined, considering specialised distributed federated learning
strategies.

## References

1.   Nicholas Carlini, Chang Liu, Jernej Kos,  ́Ulfar Erlingsson, and Dawn Song. The

secret sharer: Measuring unintended neural network memorization & extracting
secrets. CoRR, abs/1802.08232, 2018.

2.   Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential

privacy. Foundations and Trends® in Theoretical Computer Science, 9(3–4):211–
407, 2014.

3.   Vitaly Feldman. Does learning require memorization? A short tale about a long

tail. CoRR, abs/1906.05271, 2019.

4.   Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks

that exploit confidence information and basic countermeasures. In Proceedings of
the 22nd ACM SIGSAC Conference on Computer and Communications Security,

---

## Notebooks

Quick access to each notebooks, also to open on **Anaconda Notebooks**


- 1-MIA-Training - [![open_in_anaconda](https://static.anaconda.cloud/content/a22d04e8445b700f28937ab3231b8cded505d0395c63b7a269696722196d5415)](https://anaconda.cloud/api/nbserve/launch_notebook?nb_url=https%3A%2F%2Fraw.githubusercontent.com%2Fleriomaggio%2Fppml-cifris-23%2Fmain%2F1-MIA-Training.ipynb)	
- 2-MIA-Reconstruction - [![open_in_anaconda](https://static.anaconda.cloud/content/a22d04e8445b700f28937ab3231b8cded505d0395c63b7a269696722196d5415)](https://anaconda.cloud/api/nbserve/launch_notebook?nb_url=https%3A%2F%2Fraw.githubusercontent.com%2Fleriomaggio%2Fppml-cifris-23%2Fmain%2F2-MIA-Reconstruction.ipynb)
- 3-Differential Privacy in a Nutshell - [![open_in_anaconda](https://static.anaconda.cloud/content/a22d04e8445b700f28937ab3231b8cded505d0395c63b7a269696722196d5415)](https://anaconda.cloud/api/nbserve/launch_notebook?nb_url=https%3A%2F%2Fraw.githubusercontent.com%2Fleriomaggio%2Fppml-cifris-23%2Fmain%2F3-Differential-Privacy.ipynb)
- 4-MIA-DP-Training - [![open_in_anaconda](https://static.anaconda.cloud/content/a22d04e8445b700f28937ab3231b8cded505d0395c63b7a269696722196d5415)](https://anaconda.cloud/api/nbserve/launch_notebook?nb_url=https%3A%2F%2Fraw.githubusercontent.com%2Fleriomaggio%2Fppml-cifris-23%2Fmain%2F4-MIA-Reconstruction-OPACUS.ipynb)
- 5-MIA-DP-Focus - [![open_in_anaconda](https://static.anaconda.cloud/content/a22d04e8445b700f28937ab3231b8cded505d0395c63b7a269696722196d5415)](https://anaconda.cloud/api/nbserve/launch_notebook?nb_url=https%3A%2F%2Fraw.githubusercontent.com%2Fleriomaggio%2Fppml-cifris-23%2Fmain%2F5-MIA-Training-OPACUS.ipynb)

## Get the material

Clone the current repository by running the following instructions:

```bash
cd $HOME  # This will make sure you'll be in your HOME folder
git clone https://github.com/leriomaggio/ppml-cifris-23.git
```

**Note**: This will create a new folder named `ppml-cifris-23`. Move into this folder by typing:

```bash
cd ppml-cifris-23
```

Well done! Now you should do be in the right location.
Bear with me for another few seconds, following instructions reported below 🙏

## Installation Instructions (or not 🙃)

All the materials in this tutorial (code, and lecture notes) are made available as
Jupyter notebooks.

**(1)** There is no specific _hardware requirement_ to execute the code, i.e. running everything
on your laptop should be more than fine 😊.

**(2)**: As for the _software requirements_, we will be using a pretty standard Python/PyData stack:
`numpy`, `pandas`, `matplotlib`, and `scikit-learn` for all the data science and Machine learning parts,
along with `pytorch` and `torchvision` to work on the Deep Learning examples.

Moreover, a few **extra** / specialised packages will be also featured:
- [Opacus](https://opacus.ai): A library to train PyTorch models with differential privacy
- [PHE](https://pypi.org/project/phe/): A Python 3 library implementing the Paillier Partially Homomorphic Encryption
- [Flower](https://flower.dev): A Federated Learning library for PyTorch


To get ready to run the code in this tutorial you could either (a) install and configure a (`conda`) environment
on your computer with all the necessary dependency; or (b) use [**Anaconda Notebooks**](https://nb.anaconda.cloud)
and run everything without installing anything at all on your computer.

Please refer to the [`setup.md`](./setup.md) document for step-by-step instructions, or to get a special
**discount code** to access Anaconda Notebooks.

If you spot any error/mistake, please feel free to reach out directly to [me](mailto:vmaggio@anaconda.com?subject=PPML%20SciPy23%20Issue), or to open an [Issue](http://github.com/leriomaggio/ppml-tutorial/issues)
on the repository.

Any feedback will be very much appreciated!

Thank you! 🙏

## Colophon

**Author**: Valerio Maggio ([`@leriomaggio`](https://twitter.com/leriomaggio)),
Researcher, [SSI Fellow](https://www.software.ac.uk/about/fellows/valerio-maggio),
and Data Scientist Advocate at Anaconda.

All the **Code** material is distributed under the terms of the Apache License. See [LICENSE](./LICENSE) file for additional details.

All the instructional materials in this repository are free to use, and made available under the [Creative Commons Attribution
license][https://creativecommons.org/licenses/by/4.0/]. The following is a human-readable summary of (and not a substitute for) the [full legal text of the CC BY 4.0
license](https://creativecommons.org/licenses/by/4.0/legalcode).

You are free:

* to **Share**---copy and redistribute the material in any medium or format
* to **Adapt**---remix, transform, and build upon the material

for any purpose, even commercially.

The licensor cannot revoke these freedoms as long as you follow the
license terms.

Under the following terms:

* **Attribution**---You must give appropriate credit (mentioning that
  your work is derived from work that is Copyright © Software
  Carpentry and, where practical, linking to
  http://software-carpentry.org/), provide a [link to the
  license][cc-by-human], and indicate if changes were made. You may do
  so in any reasonable manner, but not in any way that suggests the
  licensor endorses you or your use.

**No additional restrictions**---You may not apply legal terms or
technological measures that legally restrict others from doing
anything the license permits.

### Acknowledgment and funding

The material developed in this tutorial has been supported by Anaconda, and the [Software Sustainability Institute](https://www.software.ac.uk) (SSI), as part of my [SSI fellowship](https://www.software.ac.uk/about/fellows/valerio-maggio) on `PETs` (Privacy Enhancing Technologies).

Please see this [deck](https://speakerdeck.com/leriomaggio/privacy-enhancing-data-science-ssi-fellowship-2022) to know more about my fellowship plans.

Public shout out to all the people at [OpenMined](https://www.openmined.org) for all the encouragement and support with the preparation of this tutorial.
I hope the material in this repository could contribute to raise awareness about all the amazing work on PETs it's being provided to the Open Source and the Python communities.

![Anaconda Logo](./logos/anaconda_logo_small.png "Anaconda")
![OpenMined](./logos/openmined_logo_small.png "OpenMined")

## Contacts

For any questions or doubts, feel free to open an [issue](https://github.com/leriomaggio/ppml-tutorial/issues) in the repository, or drop me an email @ `vmaggio_at_anaconda_dot_com`
