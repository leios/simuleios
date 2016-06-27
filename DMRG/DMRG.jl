#-------------DMRG.jl----------------------------------------------------------#
#
# Purpose: A test of a simple DMRG algorithm, following the simple DMRG guide:
#              http://simple-dmrg.readthedocs.io/en/latest/
#
#   Notes: This is done in julia because of the difficultly in the DMRG 
#          algorithm, itself.
#------------------------------------------------------------------------------#

# Compatibility necessary for later in the algorithm
using Compat

# Data structure for block and enlarged block for later
# immutable are types that are passed by copying rather than reference
# May change in the future, depending on how efficient we want to be
immutable Block
    length::Int
    basis_size::Int
    operator_dict::Dict{Symbol, AbstractMatrix{Float64}}
end

immutable EnlargedBlock
    length::Int
    basis_size::Int
    operator_dict::Dict{Symbol, AbstractMatrix{Float64}}
end

# Defining global variables
# Variable Declarations
# Specific to the XXZ chain
model_d = 2

# Single site S^z
Sz1 = [0.5 0; 0 -0.5]

# Single site S^+
Sp1 = [0 1; 0 0]

# Single site H, magnetic field
H1 = [0 0; 0 0]
#H1 = [1 1; 1 1]


# Creating initial block, conn is the connection operator for the inside edge of
# of block
initial_block = Block(1, model_d, @compat Dict{Symbol, AbstractMatrix{Float64}}(
    :H => H1,
    :conn_Sz => Sz1,
    :conn_Sp => Sp1,
))


# This is computer science magic -- Deprecated
# Basis_size must be the dimension of each operator matrix
isvalid(block::Union(Block, EnlargedBlock)) =
    # All checks to make sure all interior components are true
    all(op -> size(op) == (block.basis_size, block.basis_size), 
                          values(block.operator_dict))

# Function to combine two sites together with kron product
function site_join(Sz1, Sp1, Sz2, Sp2)
    const J = 1.0
    const Jz = 1.0
    combined_site = (J/2) * (kron(Sp1, Sp2') + kron(Sp1', Sp2)) 
                    + Jz * kron(Sz1, Sz2)
    return(combined_site)
end

# Function to enlarge provided block
function enlarge_block(block::Block)
    mblock = block.basis_size
    Op = block.operator_dict

    # Creating operators for enlarged block
    enlarged_Op_dict = @compat Dict{Symbol, AbstractMatrix{Float64}}(
        :H => kron(Op[:H], speye(model_d)) + kron(speye(mblock), H1)
              + site_join(Op[:conn_Sz], Op[:conn_Sp], Sz1, Sp1),
        :conn_Sz => kron(speye(mblock), Sz1),
        :conn_Sp => kron(speye(mblock), Sp1),
    )

    return EnlargedBlock(block.length + 1, block.basis_size * model_d,
                         enlarged_Op_dict)
end

# Function to perform a single DMRG step
function DMRG_step(sys::Block, env::Block, m::Int)

    # Checking system and environment variables are valid
    @assert isvalid(sys)
    @assert isvalid(env)

    # Enlarging each block
    sys_1 = enlarge_block(sys)
    if sys === env
        env_1 = sys_1
    else
        env_1 = enlarge_block(env)
    end

    # Checking the new system and environment
    @assert isvalid(sys_1)
    @assert isvalid(env_1)

    # Constructing the full superblock hamiltonian
    m_sys_1 = sys_1.basis_size
    m_env_1 = env_1.basis_size
    sys_1_op = sys_1.operator_dict
    env_1_op = env_1.operator_dict
    super_ham = kron(sys_1_op[:H], speye(m_env_1)) 
                + kron(speye(m_sys_1), env_1_op[:H])
                + site_join(sys_1_op[:conn_Sz], sys_1_op[:conn_Sp],
                            env_1_op[:conn_Sz], env_1_op[:conn_Sp])

    # Modification to superblock hamiltonian so it is hermitian
    super_ham = (super_ham + super_ham') / 2
    (energy,), psi0 = eigs(super_ham, nev=1, which=:SR)

    # Manipulating columns and rows sys and environment
    psi0 = transpose(reshape(psi0, (env_1.basis_size, sys_1.basis_size)))
    rho = Hermitian(psi0 * psi0')

    # Diagonalization and sorting
    fact = eigfact(rho)
    evals, evecs = fact[:values], fact[:vectors]
    permutation = sortperm(evals, rev=true)

    # Build TF matrix from m most significant eigenvectors
    min_m = min(length(evals),m)
    indices = permutation[1:min_m]
    TFmatrix = evecs[:, indices]

    truncation_error = 1 - sum(evals[indices])
    println("truncation error: ", truncation_error)

    # Rotate and truncate
    new_op_dict = Dict{Symbol, AbstractMatrix{Float64}}()
    for (name, op) in sys_1.operator_dict
        new_op_dict[name] = rotate_and_truncate(op, TFmatrix)
    end

    newblock = Block(sys_1.length, min_m, new_op_dict)

    return newblock, energy

end

# transforms the operator to the new (possibly truncated) basis with a
# provided transformation matrix
function rotate_and_truncate(Op, TFmatrix)
    return TFmatrix' * (Op * TFmatrix)
end

# function for the infinite system algorithm
function infinite_sys(L::Int, m::Int)

    # opening outputfile
    out = open("out.dat", "w+")

    block = initial_block
    i = 0
    while 2 * block.length < L
        i = block.length * 2 + 2
        println("L = ", i)
        block, energy = DMRG_step(block, block, m)
        EoL = energy / (block.length * 2)
        println("E/L = ", EoL)
        print(out, i, '\t', EoL, '\n')
        #showcompact(block.operator_dict[:conn_Sz])
        println(size(block.operator_dict[:conn_Sp]))
    end
end

infinite_sys(100, 20)

println("Everything's golden!")
