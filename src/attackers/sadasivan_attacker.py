import os
from typing import List, Optional

import numpy as np

from src.attackers.base_attacker import BaseAttacker
from src.config import AttackerConfig, AttackerGenerationConfig, MetaConfig
from src.models import HfModel
from src.server import Server

# From: https://github.com/vinusankars/Reliability-of-AI-text-detectors
# KGW LeftHash h=1; eval on untargeted incoherent text (+ 1 human example); completion models


class SadasivanAttacker(BaseAttacker):
    def __init__(self, meta_cfg: MetaConfig, attacker_cfg: AttackerConfig) -> None:
        super().__init__(meta_cfg, attacker_cfg)
        self.model = HfModel(meta_cfg, attacker_cfg.model)
        words = (
            "a, about, after, all, also, an, and, any, are, as, at, away, be, because,"
            " been, before, being, between, both, but, by, came, can, come, could, day,"
            " did, do, down, each, even, first, for, from, get, give, go, had, has,"
            " have, he, her, here, him, his, how, I, if, in, into, is, it, its, just,"
            " know, like, little, long, look, made, make, many, may, me, might,"
            " more, most, much, must, my, never, new, no, not, now, of, on, one, only,"
            " or, other, our, out, over, people, say, see, she, should, so, some, take,"
            " tell, than, that, the, their, them, then, there, these, they, thing,"
            " think, this, those, time, to, two, up, us, use, very, want,"
            " was, way, we, well, were, what, when, where, which, while, who, will,"
            " with, would, year, you, your, Time, Year, People, Way, Day, Man, Thing,"
            " Woman, Life, Child, World, School, State, Family, Student, Group, Country,"
            " Problem, Hand, Part, Place, Case, Week, Company, System,"
            " Program, Question, Work, Government, Number, Night, Point, Home, Water,"
            " Room, Mother, Area, Money, Story, Fact, Month, Lot, Right, Study, Book,"
            " Eye, Job, Word, Business, Service"
        )
        words_split = [i.strip().lower() for i in words.split(",")]
        index_to_word = {i: words[i] for i in range(len(words_split))}

        self.word_to_index = {
            index_to_word[list(index_to_word.keys())[i]]: list(index_to_word.keys())[i]
            for i in range(len(words))
        }
        self.words = words
        self.query_dir = os.path.join(self.out_root_dir, "sadasivan")

    def query_server_and_save(self, server: Server) -> None:
        N = len(self.words)
        self.M = np.zeros((N, N))
        found = 0
        thresh = 5000

        for nb_queries in range(1000000):  # will break inside
            toks = [self.words[i] for i in np.random.randint(0, len(self.words), size=100)]
            input_text = " ".join(toks)
            responses_wm, _ = server.generate([input_text])
            resp_toks = responses_wm[0].split(" ")

            for j in range(len(resp_toks) - 1):
                if resp_toks[j].lower() in self.words and resp_toks[j + 1].lower() in self.words:
                    i1 = self.word_to_index[resp_toks[j].lower()]
                    i2 = self.word_to_index[resp_toks[j + 1].lower()]
                    self.M[i1][i2] += 1
                    found += 1
            print(f"Done with {nb_queries}. {found:8d} pairs found.")

            if found >= 10**6:
                print("Done, saving")
                np.save(os.path.join(self.query_dir, "matrix.txt"), self.M)
                break

            if found >= thresh:
                print("Adding 5k more")
                np.save(os.path.join(self.query_dir, "matrix.txt"), self.M)
                thresh += 5000

    def load_queries_and_learn(self, base: bool) -> None:
        # No DoF here, just load the matrix
        self.M = np.load(os.path.join(self.query_dir, "matrix.txt"))

    def generate(
        self, prompts: List[str], cfg_gen: Optional[AttackerGenerationConfig], reseed: bool
    ) -> List[str]:
        # Ignores the prompt and just outputs text of length 200 using self.M
        lastidx = np.random.randint(0, len(self.words))
        words = [self.words[lastidx]]
        for i in range(200):
            weights = self.M[lastidx]
            lastidx = int(np.random.multinomial(1, weights / weights.sum()).argmax())
            words.append(self.words[lastidx])
        return [" ".join(words)]
