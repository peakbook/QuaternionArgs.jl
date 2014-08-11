using Quaternions
using QuaternionArgs
using Base.Test

eps = 1e-15

function test_getter()
	a = QuaternionArg(10,1,2,3)
	@assert amp(a)==10
	@assert phase1(a)==1
	@assert phase2(a)==2
	@assert phase3(a)==3

	b = QuaternionArg(Quaternion(10,1,2,3))
	@assert real(a)==10
	@assert imagi(a)==1
	@assert imagj(a)==2
	@assert imagk(a)==3
end

function test_conversion()
	a = randn(Quaternion)
	ag = QuaternionArg(a)
	aga = Quaternion(ag)

	@assert comp(a,aga,eps)
end

function test_four_arith()
	for i=1:1000
		a = randn(Quaternion)
		b = randn(Quaternion)
		ag = QuaternionArg(a)
		bg = QuaternionArg(b)

		anorm = normalize(a)
		@assert amp(anorm)==1.0

		@assert comp(a+b, Quaternion(ag+bg), eps)
		@assert comp(a-b, Quaternion(ag-bg), eps)
		@assert comp(a*b, Quaternion(ag*bg), eps)
		@assert comp(a/b, Quaternion(ag/bg), eps)
	end
end




