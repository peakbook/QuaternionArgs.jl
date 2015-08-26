module QuaternionArgs
using Quaternions
import Base: int, convert, promote_rule, show, real, imag, conj, abs, abs2, inv, +, -, /, *, rand
import Quaternions: Quaternion, imagi, imagj, imagk

export QuaternionArg, QuaternionArg256, QuaternionArg128, QuaternionArg64
export amp, phase1, phase2, phase3, normalize, comp

immutable QuaternionArg{T<:Real} <: Number
    q::T
    phi::T
    theta::T
    psi::T
end

QuaternionArg(q::Real,phi::Real,theta::Real,psi::Real) = QuaternionArg(promote(q,phi,theta,psi)...)
QuaternionArg(x::Real) = QuaternionArg(x,zero(x),zero(x),zero(x))
QuaternionArg(q::Quaternion) = convert(QuaternionArg,q)

function Quaternion(qarg::QuaternionArg)
    q0=qarg.q*(cos(qarg.phi)*cos(qarg.theta)*cos(qarg.psi) + sin(qarg.phi)*sin(qarg.theta)*sin(qarg.psi))
    q1=qarg.q*(sin(qarg.phi)*cos(qarg.theta)*cos(qarg.psi) - cos(qarg.phi)*sin(qarg.theta)*sin(qarg.psi))
    q2=qarg.q*(cos(qarg.phi)*sin(qarg.theta)*cos(qarg.psi) - sin(qarg.phi)*cos(qarg.theta)*sin(qarg.psi))
    q3=qarg.q*(cos(qarg.phi)*cos(qarg.theta)*sin(qarg.psi) + sin(qarg.phi)*sin(qarg.theta)*cos(qarg.psi))

    return Quaternion(q0,q1,q2,q3)
end

function QuaternionArg(x::Quaternion)
    q = abs(x)
    if q==zero(typeof(q))
        return QuaternionArg(zero(typeof(q)))
    end
    x = x/q

    val = 2.0*(x.q1*x.q2 - x.q0*x.q3)
    # error adjustment for satisfying the domain of asin
    if val > one(typeof(q))
        val = one(typeof(q))
    elseif val < -one(typeof(q))
        val = -one(typeof(q))
    end
    psi = -0.5*asin(val)

    if (psi != pi/4) && (psi != -pi/4)
        phi  = 0.5*argi(x*beta((conj(x))))
        theta= 0.5*argj(alpha(conj(x))*x)
    else 
        phi = 0
        theta= 0.5*argj(gamma(conj(x))*x)
    end

    t = Quaternion(QuaternionArg(1.0,phi,theta,psi));
    if comp(t,-x)
        phi = phi - sign(phi)*pi
    end

    return QuaternionArg(q, phi, theta, psi)
end

typealias QuaternionArg256 QuaternionArg{Float64}
typealias QuaternionArg128 QuaternionArg{Float32}
typealias QuaternionArg64 QuaternionArg{Float16}

convert(::Type{QuaternionArg}, x::Real) = QuaternionArg(x)
convert{T<:Real}(::Type{QuaternionArg{T}}, x::Real) = QuaternionArg(x)
convert{T<:Real}(::Type{QuaternionArg{T}}, q::QuaternionArg) = QuaternionArg{T}(convert(T,q.q), convert(T,q.phi), convert(T,q.theta), convert(T,q.psi))
convert{T<:Real}(::Type{QuaternionArg{T}}, q::QuaternionArg{T}) = q
convert{T<:Real}(::Type{QuaternionArg{T}}, q::Quaternion{T}) = QuaternionArg(q)
convert{T<:Real}(::Type{Quaternion{T}}, q::QuaternionArg{T}) = Quaternion(q)

promote_rule{T<:Real}(::Type{QuaternionArg{T}}, ::Type{T}) = QuaternionArg{T}
promote_rule{T<:Real}(::Type{QuaternionArg}, ::Type{T}) = QuaternionArg
promote_rule{T<:Real,S<:Real}(::Type{QuaternionArg{T}}, ::Type{S}) = QuaternionArg{promote_type(T,S)}
promote_rule{T<:Real,S<:Real}(::Type{QuaternionArg{T}}, ::Type{QuaternionArg{S}}) = QuaternionArg{promote_type(T,S)}
promote_rule{T<:Real,S<:Real}(::Type{QuaternionArg{T}}, ::Type{Quaternion{S}}) = QuaternionArg{promote_type(T,S)}
promote_rule{T<:Real,S<:Real}(::Type{Quaternion{T}}, ::Type{QuaternionArg{S}}) = Quaternion{promote_type(T,S)}

quaternionArg(q,phi,theta,psi) = QuaternionArg(q,phi,theta,psi)
quaternionArg(x) = QuaternionArg(x)
quaternionArg(q::QuaternionArg) = q 

quaternionArg256(q::Float64,phi::Float64,theta::Float64,psi::Float64) = QuaternionArg{Float64}(q,phi,theta,psi)
quaternionArg256(q::Real,phi::Real,theta::Real,psi::Real) = quaternion256(float64(q),float64(phi),float64(theta),float64(psi))
quaternionArg128(q::Float32,phi::Float32,theta::Float32,psi::Float32) = QuaternionArg{Float32}(q,phi,theta,psi)
quaternionArg128(q::Real,phi::Real,theta::Real,psi::Real) = quaternion128(float32(q),float32(phi),float32(theta),float32(psi))
quaternionArg64(q::Float16,phi::Float16,theta::Float16,psi::Float16) = QuaternionArg{Float16}(q,phi,theta,psi)
quaternionArg64(q::Real,phi::Real,theta::Real,psi::Real) = quaternion64(float16(q),float16(phi),float16(theta),float16(psi))

#for fn in _numeric_conversion_func_names
for fn in (:int,:integer,:signed,:int8,:int16,:int32,:int64,:int128,
    :uint,:unsigned,:uint8,:uint16,:uint32,:uint64,:uint128,
    :float,:float16,:float32,:float64)
    @eval $fn(q::QuaternionArg) = QuaternionArg($fn(q.q),$fn(q.phi),$fn(q.theta),$fn(q.psi))
end

function show(io::IO, z::QuaternionArg)
    pm(z) = z < 0 ? "-$(-z)" : "+$z"
    print(io, z.q, " (phi=",pm(z.phi)," theta=", pm(z.theta), " psi=", pm(z.psi), ")")
end

for (f,t) in ((:quaternionArg64, QuaternionArg64),
    (:quaternionArg128, QuaternionArg128),
    (:quaternionArg256, QuaternionArg256))
    @eval ($f)(x::AbstractArray{$t}) = x
    @eval ($f)(x::AbstractArray) = copy!(similar(x,$t), x)
end

QuaternionArg{T<:QuaternionArg}(x::AbstractArray{T}) = x
QuaternionArg(x::AbstractArray) = copy!(similar(x,typeof(quaternion(one(eltype(x))))), x)
QuaternionArg{T<:Quaternion}(x::AbstractArray{T}) = map(QuaternionArg,x)
Quaternion{T<:QuaternionArg}(x::AbstractArray{T}) = map(Quaternion,x)

amp(z::QuaternionArg) = z.q
phase1(z::QuaternionArg) = z.phi
phase2(z::QuaternionArg) = z.theta
phase3(z::QuaternionArg) = z.psi
normalize(z::QuaternionArg) = QuaternionArg(1.0,z.phi,z.theta,z.psi)

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

rand{T<:Real}(::Type{QuaternionArg{T}}) = quaternionArg(one(T),convert(T,2.0)*pi*(rand(T)-convert(T,0.5)),one(T)*pi*(rand(T)-convert(T,0.5)),convert(T,0.5)*pi*(rand(T)-convert(T,0.5)))

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
    if abs(x-y) > eps(max(x,y)*1E4)
        return false
    end
    return true
end

function comp(a::Quaternion, b::Quaternion)
    flg = comp(a.q0, b.q0) &
    comp(a.q1, b.q1) &
    comp(a.q2, b.q2) &
    comp(a.q3, b.q3)

    return flg
end

end

