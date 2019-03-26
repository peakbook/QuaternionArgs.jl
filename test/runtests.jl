using Quaternions
using LinearAlgebra
using QuaternionArgs
using Test

const N=100
@testset "QuaternionArgs Tests" begin
    @testset "getter" begin
        for typ in [QuaternionArgF16, QuaternionArgF32, QuaternionArgF64]
            a = rand(typ)
            @test amp(a)==a.q
            @test phase1(a)==a.phi
            @test phase2(a)==a.theta
            @test phase3(a)==a.psi
        end
    end

    for typ in [QuaternionF16, QuaternionF32, QuaternionF64]
        @testset "$typ" begin
            @testset "conversion" begin
                for i in 1:N
                    a = QuaternionArg(randn(typ))
                    @test isapprox(a,Quaternion(a))
                end
            end

            @testset "four arithmetic" begin
                for i in 1:N
                    a = randn(typ)
                    b = randn(typ)
                    ag = QuaternionArg(a)
                    bg = QuaternionArg(b)

                    anorm = normalize(ag)
                    @test amp(anorm)==one(anorm.q)
                    @test isapprox(a+b, Quaternion(ag+bg))
                    @test isapprox(a-b, Quaternion(ag-bg))
                    @test isapprox(a*b, Quaternion(ag*bg))
                    @test isapprox(a/b, Quaternion(ag/bg))
                end
            end
        end
    end
end
