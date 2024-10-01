using Sbi
using Test
using StableRNGs
using Lux
using Random


# Note for MaskedLinear Layer, the use bias keyword is not implemented, I don't know why it would be needed
# versus the set bias to zeros but this is the situation for now
@testset "Sbi.jl" begin
    # Write your tests here.
    rng = StableRNG(12345)

    @testset "constructors" begin
        layer = MaskedLinear(10, 100)
        ps, st = Lux.setup(rng, layer)

        @test size(ps.weight) == (100, 10)
        @test size(ps.bias) == (100, 1)  #Lux uses (100, ) why?
        @test layer.activation == identity

    end


    @testset "dimensions" begin
            layer = MaskedLinear(10, 5)
            ps, st = Lux.setup(rng, layer)

            @test size(first(Lux.apply(layer, randn(10), ps, st))) == (5, 1)
            @test size(first(Lux.apply(layer, randn(10, 2), ps, st))) == (5, 2)

            @test LuxCore.outputsize(layer, randn(10), rng) == (5,)
    end


    @testset "zeros" begin
            @test begin
                layer = MaskedLinear(10, 1, identity; init_weight=ones32, init_bias=zeros32)
                first(Lux.apply(
                    layer, ones(10, 1), Lux.setup(rng, layer)...))
            end == 10 * ones(1, 1)

            @test begin
                layer = MaskedLinear(10, 1, identity; init_weight=ones32, init_bias=zeros32)
                first(Lux.apply(
                    layer, ones(10, 2) , Lux.setup(rng, layer)...))
            end == 10 * ones(1, 2)

            @test begin
                layer = MaskedLinear(10, 2, identity; init_weight=ones32, init_bias=zeros32)
                first(Lux.apply(
                    layer, ones(10, 1) , Lux.setup(rng, layer)...))
            end == 10 * ones(2, 1)

            @test begin
                layer = MaskedLinear(10, 2, identity; init_weight=ones32, init_bias=zeros32)
                first(Lux.apply(layer, [ones(10, 1) 2 * ones(10, 1)],
                    Lux.setup(rng, layer)...))
            end == [10 20; 10 20]

    end

    @testset "Mask" begin
        # case where first layer

        layer = MaskedLinear(10, 100, identity; init_weight=ones32, init_bias=zeros32)
        ps, st = Lux.setup(rng, layer)

        #case wher last layer


        # Function to create a row with exactly 5 spots set to 1.0 and 5 spots set to 0.0 (as Float32)
        function create_row()
            row = [1.0f0 for _ in 1:5]  # Create 5 ones as Float32
            append!(row, [0.0f0 for _ in 1:5])  # Append 5 zeros as Float32
            shuffle!(row)  # Shuffle the row to randomize the positions
            return row
        end

        # Create the matrix with 100 rows and 10 columns, ensuring Float32 type
        matrix = [create_row() for _ in 1:100]

        # Convert the array of arrays into a matrix directly with Float32 elements
        matrix = hcat(matrix...)'  # Transpose after hcat to ensure proper dimensions

        # Convert the matrix to Float32 (though it already should be in Float32)
        matrix = Float32.(matrix)

        # Set Layer Mask to this value

        layer.init_mask[] = matrix

        @test first(Lux.apply(layer, ones(10, ), ps, st)) == 5*ones(100, 1)

    end


end
