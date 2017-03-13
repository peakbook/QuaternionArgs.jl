__precompile__()

module QuaternionArgs
using Quaternions
using Compat
import Base: normalize, convert, promote_rule, show, real, imag, conj, abs, abs2, inv, +, -, /, *, rand
import Quaternions: Quaternion, imagi, imagj, imagk

export QuaternionArg, QuaternionArg256, QuaternionArg128, QuaternionArg64
export amp, phase1, phase2, phase3

const tor = 1E6

immutable QuaternionArg{T<:AbstractFloat} <: Number
    q::T
    phi::T
    theta::T
    psi::T
end

QuaternionArg(q::AbstractFloat,phi::AbstractFloat,theta::AbstractFloat,psi::AbstractFloat) = QuaternionArg(promote(q,phi,theta,psi)...)
QuaternionArg(q::Integer,phi::Integer,theta::Integer,psi::Integer) = QuaternionArg{Float64}(promote(q,phi,theta,psi)...)
QuaternionArg(x::AbstractFloat) = QuaternionArg(x,zero(x),zero(x),zero(x))
QuaternionArg(x::Integer) = QuaternionArg(float(x))

function Quaternion(qarg::QuaternionArg)
    q0=qarg.q*(cos(qarg.phi)*cos(qarg.theta)*cos(qarg.psi) + sin(qarg.phi)*sin(qarg.theta)*sin(qarg.psi))
    q1=qarg.q*(sin(qarg.phi)*cos(qarg.theta)*cos(qarg.psi) - cos(qarg.phi)*sin(qarg.theta)*sin(qarg.psi))
    q2=qarg.q*(cos(qarg.phi)*sin(qarg.theta)*cos(qarg.psi) - sin(qarg.phi)*cos(qarg.theta)*sin(qarg.psi))
    q3=qarg.q*(cos(qarg.phi)*cos(qarg.theta)*sin(qarg.psi) + sin(qarg.phi)*sin(qarg.theta)*cos(qarg.psi))

    return Quaternion(q0,q1,q2,q3)
end

function QuaternionArg{T<:Real}(x::Quaternion{T})
    QuaternionArg(convert(Quaternion256,x))
end

function QuaternionArg{T<:AbstractFloat}(x::Quaternion{T})
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
    if comp(t,-x)
        phi = phi - sign(phi)*T(pi)
    end

    return QuaternionArg(q, phi, theta, psi)
end

@compat const QuaternionArg256 = QuaternionArg{Float64}
@compat const QuaternionArg128 = QuaternionArg{Float32}
@compat const QuaternionArg64 = QuaternionArg{Float16}

convert(::Type{QuaternionArg}, x::Real) = QuaternionArg(x)
convert{T<:AbstractFloat}(::Type{QuaternionArg{T}}, x::Real) = QuaternionArg(x)
convert{T<:AbstractFloat,S<:AbstractFloat}(::Type{QuaternionArg{T}}, q::QuaternionArg{S}) = QuaternionArg{T}(T(q.q), T(q.phi), T(q.theta), T(q.psi))
convert{T<:AbstractFloat}(::Type{QuaternionArg{T}}, q::QuaternionArg{T}) = q
convert{T<:AbstractFloat}(::Type{QuaternionArg{T}}, q::Quaternion{T}) = QuaternionArg(q)
convert{T<:Real}(::Type{Quaternion{T}}, q::QuaternionArg{T}) = Quaternion(q)

promote_rule{T<:AbstractFloat,S<:Real}(::Type{QuaternionArg{T}}, ::Type{S}) = QuaternionArg{promote_type(T,S)}
promote_rule{T<:AbstractFloat,S<:Real}(::Type{QuaternionArg{T}}, ::Type{QuaternionArg{S}}) = QuaternionArg{promote_type(T,S)}
promote_rule{T<:AbstractFloat,S<:Real}(::Type{QuaternionArg{T}}, ::Type{Quaternion{S}}) = Quaternion{promote_type(T,S)}

quaternionArg(q,phi,theta,psi) = QuaternionArg(q,phi,theta,psi)
quaternionArg(x) = QuaternionArg(x)
quaternionArg(q::QuaternionArg) = q 

function show(io::IO, z::QuaternionArg)
    pm(z) = z < 0 ? "-$(-z)" : "+$z"
    print(io, z.q, " (phi=",pm(z.phi)," theta=", pm(z.theta), " psi=", pm(z.psi), ")")
end

QuaternionArg{T<:QuaternionArg}(x::AbstractArray{T}) = x
QuaternionArg(x::AbstractArray) = copy!(similar(x,typeof(quaternion(one(eltype(x))))), x)
QuaternionArg{T<:Quaternion}(x::AbstractArray{T}) = map(QuaternionArg,x)
Quaternion{T<:QuaternionArg}(x::AbstractArray{T}) = map(Quaternion,x)

amp(z::QuaternionArg) = z.q
phase1(z::QuaternionArg) = z.phi
phase2(z::QuaternionArg) = z.theta
phase3(z::QuaternionArg) = z.psi
normalize(z::QuaternionArg) = QuaternionArg(one(z.q),z.phi,z.theta,z.psi)

phase1{T<:Real}(x::AbstractVector{T}) = zero(x)
phase2{T<:Real}(x::AbstractVector{T}) = zero(x)
phase3{T<:Real}(x::AbstractVector{T}) = zero(x)

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

rand{T<:AbstractFloat}(::Type{QuaternionArg{T}}) = quaternionArg(one(T),T(2.0)*T(pi)*(rand(T)-T(0.5)),one(T)*T(pi)*(rand(T)-T(0.5)),T(0.5)*T(pi)*(rand(T)-T(0.5)))

real(z::QuaternionArg)  = Quaternion(z).q0
imagi(z::QuaternionArg) = Quaternion(z).q1
imagj(z::QuaternionArg) = Quaternion(z).q2
imagk(z::QuaternionArg) = Quaternion(z).q3

argi(z::Quaternion) = atan2(z.q1,z.q0)
argj(z::Quaternion) = atan2(z.q2,z.q0)
argk(z::Quaternion) = atan2(z.q3,z.q0)

alpha(z::Quaternion) = Quaternion(z.q0,  z.q1, -z.q2, -z.q3)
beta(z::Quaternion)  = Quaternion(z.q0, -z.q1,  z.q2, -z.q3)
gamma(z::Quaternion) = Quaternion(z.q0, -z.q1, -z.q2,  z.q3)

function comp(x::Real, y::Real)
    return abs(x-y) <= eps(max(x,y)*tor)
end

function comp(a::Quaternion, b::Quaternion)
    flg = comp(a.q0, b.q0) &
    comp(a.q1, b.q1) &
    comp(a.q2, b.q2) &
    comp(a.q3, b.q3)

    return flg
end

end

