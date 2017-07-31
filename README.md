# Curriculum Dropout

## The idea
[Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) is a very effective way of regularizing neural networks. Stochastically "dropping out" units with a certain probability discourages over-specific co-adaptations of feature detectors, preventing overfitting and improving network generalization. However, we show that using a fixed dropout probability during training is a suboptimal choice. We propose a time scheduling for the probability of retaining neurons in the network. This induces an adaptive regularization scheme that smoothly increases the difficulty of the optimization problem. This idea of "starting easy" and adaptively increasing the difficulty of the learning problem has its roots in [curriculum learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf) and allows one to train better models. Indeed, we prove that our optimization strategy implements a very general curriculum scheme, by gradually adding noise to both the input and intermediate feature representations in the network architecture. The method, named **Curriculum Dropout**, yields to better generalization.

## Reference

If you use this code as part of any published research, please acknowledge the following paper:

**"Curriculum Dropout"**  
Pietro Morerio, Jacopo Cavazza, Riccardo Volpi, René Vidal and Vittorio Murino *[arXiv](https://arxiv.org/abs/1703.06229)*

    @InProceedings{Morerio2017dropout,
        title={Curriculum Dropout},
        author={Morerio, Pietro and Cavazza, Jacopo and Volpi, Riccardo and Vidal, Ren\'e and Murino, Vittorio},
        booktitle = {ICCV},
        year={2017}
    } 

## License
This repsoitory is released under the GNU GENERAL PUBLIC LICENSE.
