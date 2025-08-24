import time


# Generate all monomials of a given degree (up to `degree`)
def degree_d_monomials(F, Fxx, num_vars, degree):
    if degree == 0:
        return [F(1)]
    prev_degree_monomials = degree_d_monomials(F, Fxx, num_vars, degree - 1)
    cur_degree_monomials = []
    for monomial in prev_degree_monomials: 
        for var in Fxx.gens():
            cur_degree_monomials += [var * monomial]
    cur_degree_monomials += prev_degree_monomials   
    cur_degree_monomials = list(set(cur_degree_monomials))  
    return cur_degree_monomials

# Generate random polynomials of given degree `degree`
def generate_degree_d_polynomials(F, Fxx, num_vars, num_polys, degree):
    system = []
    monomials = degree_d_monomials(F, Fxx, num_vars, degree)
    for i in range(num_polys):
        poly = sum(F.random_element() * monomial for monomial in monomials)
        system += [poly]
    return system


def compute_binomial(num_vars, monomial_degree, polynomial_degree ):
    return binomial(num_vars + monomial_degree + polynomial_degree, monomial_degree + polynomial_degree)

def main():
    # Parameters
    num_vars = 7            # Number of variables in the polynomial ring
    monomial_degree = 3     # Degree of monomials used for system extension
    polynomial_degree = 8   # Degree of the randomly generated polynomials
    num_polys = 1000        # Number of random polynomials to generate

    F = GF(31)

    Fxx = PolynomialRing(F, 'x', num_vars)
    x = Fxx.gens()

    monomials = degree_d_monomials(F, Fxx, num_vars, monomial_degree)
    polynomials = generate_degree_d_polynomials(F, Fxx, num_vars, num_polys, polynomial_degree)

    k = compute_binomial(num_vars, monomial_degree, polynomial_degree)
    count = 0
    extended_system = []
    for poly in polynomials:
        for mono in monomials: 
            extended_system += [poly * mono]
            count += 1
            if count > k:
                break
        if count > k:
            break

    I = ideal(extended_system)
    polynomials_totalnumber = len(monomials) * num_polys
    Fseq = Sequence(extended_system)
    A,v = Fseq.coefficient_matrix()

    print(f"binomial is {k}")
    print(f"|A| is {polynomials_totalnumber}")
    print(f"Dimension of A is {A.dimensions()} and rank of A is { A.rank()}")

if __name__ == "__main__":
    main()
