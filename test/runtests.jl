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
	@assert comp(real(a),10.0,eps)
	@assert comp(imagi(a),1.0,eps)
	@assert comp(imagj(a),2.0,eps)
	@assert comp(imagk(a),3.0,eps)
end

function test_conversion()
	a = randn(Quaternion128)
	ag = QuaternionArg(a)
	aga = Quaternion(ag)

	@assert comp(a,aga,eps)
end

function test_four_arith()
	for i=1:10000
		a = randn(Quaternion128)
		b = randn(Quaternion128)
		ag = QuaternionArg(a)
		bg = QuaternionArg(b)

		anorm = normalize(ag)
		@assert amp(anorm)==one(anorm.q)

		@assert comp(a+b, Quaternion(ag+bg), eps)
		@assert comp(a-b, Quaternion(ag-bg), eps)
		@assert comp(a*b, Quaternion(ag*bg), eps)
		@assert comp(a/b, Quaternion(ag/bg), eps)
	end
end




