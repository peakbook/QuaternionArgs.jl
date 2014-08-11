using Quaternions
using QuaternionArgs
using Base.Test

eps = 1e-15

function test_getter()
	a = randn(QuaternionArg128)
	@assert amp(a)==a.q
	@assert phase1(a)==a.phi
	@assert phase2(a)==a.theta
	@assert phase3(a)==a.psi

	a = normalize(randn(QuaternionArg128))
	b = Quaternion(a)
	ab = QuaternionArg(b)
	@assert comp(real(ab),b.q0,eps)
	@assert comp(imagi(ab),b.q1,eps)
	@assert comp(imagj(ab),b.q2,eps)
	@assert comp(imagk(ab),b.q3,eps)
end

function test_conversion()
	a = randn(Quaternion128)
	ag = QuaternionArg(a)
	aga = Quaternion(ag)

	@assert comp(a,aga,eps)
end

function test_four_arith()
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




