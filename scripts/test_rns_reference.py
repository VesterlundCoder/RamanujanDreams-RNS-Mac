"""CPU-only exact semantics and capacity tests for the Meijer RNS helpers."""
from pathlib import Path
import sys
import unittest
import math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'meijer_cmf'))
import rns_reference as r

class RnsReference(unittest.TestCase):
    def test_walk_zero_checkpoint_and_independent_integer_recurrence(self):
        # M(t) = [[t,1],[1,0]], including a checkpoint at depth zero.
        coeffs=[[[1,0],[0,1]],[[0,1],[0,0]]]
        ps=[101,103]
        snaps=r.walk_rns(r.encode_residues(coeffs,ps),ps,2,8,[0,1,4,8])
        P=[[1,0],[0,1]]
        for t in range(9):
            if t in snaps:
                for k,p in enumerate(ps):
                    np.testing.assert_array_equal(snaps[t][k],[[v%p for v in row] for row in P])
            P=[[row[0]*t+row[1],row[0]] for row in P]

    def test_maximum_residues_reduce_each_product(self):
        ps=[2147483647,2147483629];m=8
        coefficients=np.array([np.full((m*m,1),p-1) for p in ps],dtype=np.uint64)
        snaps=r.walk_rns(coefficients,ps,m,2,[2])
        np.testing.assert_array_equal(snaps[2],np.full((2,m,m),8))

    def test_signed_crt_odd_and_even_products(self):
        for ps in ([5,7],[2,5,7]):
            M=math.prod(ps)
            for x in range(M):
                self.assertEqual(r.garner_crt([x%p for p in ps],ps),x if 2*x<M else x-M)

    def test_integer_capacity_bound_covers_actual_growth(self):
        coeffs=[[[7,-3],[0,2]],[[-2,1],[1,1]]]
        depth=50;K=r.required_primes(coeffs,2,1,depth,margin=0);ps=r.gen_primes(K)
        P=[[1,0],[0,1]]
        for t in range(depth):
            M=[[7*t-3,2],[-2*t+1,t+1]]
            P=[[sum(P[i][q]*M[q][j] for q in range(2)) for j in range(2)] for i in range(2)]
        self.assertGreater(math.prod(ps),2*max(abs(v) for row in P for v in row))

    def test_prefix_agreement_does_not_prove_capacity(self):
        ps=[101,103,107,109];M=math.prod(ps)
        # The unseen integer is M+1; all lanes encode 1, so dropping lanes agrees.
        snap=np.ones((4,1,1),dtype=np.uint32)
        consistent,value=r.crt_capacity_ok(snap,ps,1)
        self.assertTrue(consistent)
        self.assertEqual(value,[1])
        self.assertNotEqual(value[0],M+1)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):r.gen_primes(0)
        with self.assertRaises(ValueError):r.garner_crt([1],[101,103])
        with self.assertRaises(ValueError):r.garner_crt([101],[101])
        with self.assertRaises(ValueError):r.walk_rns(np.zeros((1,1,1)),[105],1,1,[1])

if __name__=='__main__':unittest.main()
