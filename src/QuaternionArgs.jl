__precompile__()

module QuaternionArgs
using Quaternions
import Base: convert, promote_rule, show, real, imag, conj, abs, abs2, inv, +, -, /, *, rand
import Quaternions: Quaternion, imagi, imagj, imagk
import LinearAlgebra: normalize
import Random: AbstractRNG, SamplerType
import Base: isinteger, isfinite, isnan, isinf, iszero, isone

export QuaternionArg, QuaternionArgF64, QuaternionArgF32, QuaternionArgF16
export amp, phase1, phase2, phase3, normalize

struct QuaternionArg{T<:AbstractFloat} <: Number
    q::T
    phi::T
    theta::T
    psi::T
end

QuaternionArg(q::AbstractFloat,phi::AbstractFloat,theta::AbstractFloat,psi::AbstractFloat) = QuaternionArg(promote(q,phi,theta,psi)...)
QuaternionArg(q::Integer,phi::Integer,theta::Integer,psi::Integer) = QuaternionArg{Float64}(promote(q,phi,theta,psi)...)
QuaternionArg(x::AbstractFloat) = QuaternionArg(x,zero(x),zero(x),zero(x))
QuaternionArg(x::Integer) = QuaternionArg(float(x))

function Quaternion(qarg::QuaternionArg{T}) where T <: AbstractFloat
    q0=qarg.q*(cos(qarg.phi)*cos(qarg.theta)*cos(qarg.psi) + sin(qarg.phi)*sin(qarg.theta)*sin(qarg.psi))
    q1=qarg.q*(sin(qarg.phi)*cos(qarg.theta)*cos(qarg.psi) - cos(qarg.phi)*sin(qarg.theta)*sin(qarg.psi))
    q2=qarg.q*(cos(qarg.phi)*sin(qarg.theta)*cos(qarg.psi) - sin(qarg.phi)*cos(qarg.theta)*sin(qarg.psi))
    q3=qarg.q*(cos(qarg.phi)*cos(qarg.theta)*sin(qarg.psi) + sin(qarg.phi)*sin(qarg.theta)*cos(qarg.psi))

    return Quaternion(q0,q1,q2,q3)
end

function QuaternionArg(x::Quaternion{T}) where T<:Real
    QuaternionArg(convert(QuaternionF64,x))
end

function QuaternionArg(x::Quaternion{T}) where T<:AbstractFloat
    q = abs(x)
    if q==zero(typeof(q))
        return QuaternionArg(zero(typeof(q)))
    end
    x = x/q

    val = 2*(x.q1*x.q2 - x.q0*x.q3)
    # error adjustment for satisfying the domain of asin
    if val > one(typeof(q))
        val = one(typeof(q))
    elseif val < -one(typeof(q))
        val = -one(typeof(q))
    end
    psi = -asin(val)/2

    if (psi != T(pi/4)) && (psi != -T(pi/4))
        phi  = argi(x*beta((conj(x))))/2
        theta= argj(alpha(conj(x))*x)/2
    else
        phi = T(0)
        theta= argj(gamma(conj(x))*x)/2
    end

    t = Quaternion(QuaternionArg(one(T),phi,theta,psi));
    if isapprox(t,-x)
        phi = phi - sign(phi)*T(pi)
    end

    return QuaternionArg(q, phi, theta, psi)
end

const QuaternionArgF64 = QuaternionArg{Float64}
const QuaternionArgF32 = QuaternionArg{Float32}
const QuaternionArgF16 = QuaternionArg{Float16}

convert(::Type{QuaternionArg}, x::Real) = QuaternionArg(x)
convert(::Type{QuaternionArg{T}}, x::Real) where T<:AbstractFloat = QuaternionArg(x)
convert(::Type{QuaternionArg{T}}, q::QuaternionArg{S}) where {T<:AbstractFloat,S<:AbstractFloat} = QuaternionArg{T}(T(q.q), T(q.phi), T(q.theta), T(q.psi))
convert(::Type{QuaternionArg{T}}, q::QuaternionArg{T}) where T<:AbstractFloat = q
convert(::Type{QuaternionArg{T}}, q::Quaternion{T}) where T<:AbstractFloat = QuaternionArg(q)
convert(::Type{Quaternion{T}}, q::QuaternionArg{T}) where T<:Real = Quaternion(q)

promote_rule(::Type{QuaternionArg{T}}, ::Type{S}) where {T<:AbstractFloat,S<:Real} = QuaternionArg{promote_type(T,S)}
promote_rule(::Type{QuaternionArg{T}}, ::Type{QuaternionArg{S}}) where {T<:AbstractFloat,S<:Real} = QuaternionArg{promote_type(T,S)}
promote_rule(::Type{QuaternionArg{T}}, ::Type{Quaternion{S}}) where {T<:AbstractFloat,S<:Real} = Quaternion{promote_type(T,S)}

quaternionArg(q,phi,theta,psi) = QuaternionArg(q,phi,theta,psi)
quaternionArg(x) = QuaternionArg(x)
quaternionArg(q::QuaternionArg) = q

function show(io::IO, z::QuaternionArg)
    pm(z) = z < 0 ? "-$(-z)" : "+$z"
    print(io, z.q, " (phi=",pm(z.phi)," theta=", pm(z.theta), " psi=", pm(z.psi), ")")
end

QuaternionArg(x::AbstractArray{T}) where T<:QuaternionArg = x
QuaternionArg(x::AbstractArray) = copy!(similar(x,typeof(quaternion(one(eltype(x))))), x)
QuaternionArg(x::AbstractArray{T}) where T<:Quaternion = map(QuaternionArg,x)
Quaternion(x::AbstractArray{T}) where T<:QuaternionArg = map(Quaternion,x)

amp(z::QuaternionArg) = z.q
phase1(z::QuaternionArg) = z.phi
phase2(z::QuaternionArg) = z.theta
phase3(z::QuaternionArg) = z.psi
normalize(z::QuaternionArg) = QuaternionArg(one(z.q),z.phi,z.theta,z.psi)

phase1(x::AbstractVector{T}) where T<:Real = zero(x)
phase2(x::AbstractVector{T}) where T<:Real = zero(x)
phase3(x::AbstractVector{T}) where T<:Real = zero(x)

for fn in (:amp,:phase1,:phase2,:phase3,:normalize)
    @eval begin
        ($fn)(A::AbstractArray) = map(($fn),A)
    end
end

conj(z::QuaternionArg) = QuaternionArg(z.q, -z.phi, -z.theta, -z.psi)
abs(z::QuaternionArg) = z.q
abs2(z::QuaternionArg) = z.q*z.q
inv(z::QuaternionArg) = QuaternionArg(inv(Quaternion(z)))

(-)(z::QuaternionArg) = QuaternionArg(-z.q, z.phi, z.theta, z.psi)
(/)(z::QuaternionArg, x::Real) = QuaternionArg(z.q/x, z.phi, z.theta, z.psi)
(+)(z::QuaternionArg, w::QuaternionArg) = QuaternionArg(Quaternion(z)+Quaternion(w))
(-)(z::QuaternionArg, w::QuaternionArg) = QuaternionArg(Quaternion(z)-Quaternion(w))
(*)(z::QuaternionArg, w::QuaternionArg) = QuaternionArg(Quaternion(z)*Quaternion(w))
(/)(z::QuaternionArg, w::QuaternionArg) = QuaternionArg(Quaternion(z)/Quaternion(w))

rand(r::AbstractRNG, ::SamplerType{QuaternionArg{T}}) where T<:AbstractFloat =
     quaternionArg(one(T),T(2.0)*T(pi)*(rand(T)-T(0.5)),one(T)*T(pi)*(rand(T)-T(0.5)),T(0.5)*T(pi)*(rand(T)-T(0.5)))

real(z::QuaternionArg)  = Quaternion(z).q0
imagi(z::QuaternionArg) = Quaternion(z).q1
imagj(z::QuaternionArg) = Quaternion(z).q2
imagk(z::QuaternionArg) = Quaternion(z).q3

argi(z::Quaternion) = atan(z.q1,z.q0)
argj(z::Quaternion) = atan(z.q2,z.q0)
argk(z::Quaternion) = atan(z.q3,z.q0)

alpha(z::Quaternion) = Quaternion(z.q0,  z.q1, -z.q2, -z.q3)
beta(z::Quaternion)  = Quaternion(z.q0, -z.q1,  z.q2, -z.q3)
gamma(z::Quaternion) = Quaternion(z.q0, -z.q1, -z.q2,  z.q3)

isreal(q::QuaternionArg) = isreal(Quaternion(q))
isinteger(q::QuaternionArg) = isinteger(Quaternion(q))
isfinite(q::QuaternionArg) = isfinite(Quaternion(q))
isnan(q::QuaternionArg) = isnan(Quaternion(q))
isinf(q::QuaternionArg) = isinf(Quaternion(q))
iszero(q::QuaternionArg) = iszero(Quaternion(q))
isone(q::QuaternionArg) = isone(Quaternion(q))

end
