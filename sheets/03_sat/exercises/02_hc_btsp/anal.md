in hamiltonian file, i used to  add a new clause to the sat to ensure that at least there one edge leaving s
which was okay, it passed all the tests, but when it got called on the second instance of btsp, we got a runtime issue
        if existing_edges:
            self.solver.add_clause(existing_edges)
sooo, finally i kept track on the existing edges, negated them, and said at most we have len(edges) - 1, which is equivalent to say, at least one edge is leaving, which made then btps also pass the tests

    neg_edges = [-x for x in edges]
    self.solver.add_atmost(neg_edges, len(edges) - 1)