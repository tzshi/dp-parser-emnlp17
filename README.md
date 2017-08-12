# dp-parser-emnlp17

This is the repo for the code to reproduce the experiments and results of our EMNLP 2017 paper.

For more up-to-date parser under maintenance, please see <https://github.com/tzshi/cdparser>.

See also our system entry [C2L2](https://github.com/CoNLL-UD-2017/C2L2) to the CoNLL 2017 Shared Task on parsing Universal Dependencies, which applied the same algorithms as described in the EMNLP paper.
The system won second place in overall evaluation, and the first place in the category of surprise langugaes.

# Requirements

- [DyNet](https://github.com/clab/dynet) 1.1

# Documentation

More to come.

# Citation

If you make use of this software in your research, we appreciate you citing the following paper:

```
@InProceedings{shi+huang+lee2017exact,
    author    = {Shi, Tianze  and  Huang, Liang  and  Lee, Lillian},
    title     = {Fast(er) Exact Decoding and Global Training for Transition-based Dependency Parsing via a Minimal Feature Set},
    booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing},
    month     = {September},
    year      = {2017},
    address   = {Copenhagen, Denmark},
    publisher = {Association for Computational Linguistics},
    pages     = {(To appear)}
}
```

# Acknowledgement

When implementing the first-order graph-based algorithm, we referenced the BiST parser: <https://github.com/elikip/bist-parser>.
