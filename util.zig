const std = @import("std");
usingnamespace std.simd;

pub fn AsVec(comptime T: type) type {
    return switch(comptime @typeInfo(T)) {
        .Array, => |i| @Vector(i.len, i.child),
        .Vector, => |i| @Vector(i.len, i.child),
        .Pointer => |i| switch(comptime @typeInfo(i.child)) {
            .Pointer, .Array => AsVec(i.child),
            else => @compileError("Invalid type `" ++ @typeName(T) ++ "` passed to AsVec"),
        },
        else => @compileError("Invalid type `" ++ @typeName(T) ++ "` passed to AsVec")
    };
}

pub fn asVec(v: anytype) AsVec(@TypeOf(v)) {
    switch(comptime @typeInfo(@TypeOf(v))) {
        .Array => return v,
        .Vector => return v,
        .Pointer => |i| {
            switch(comptime @typeInfo(i.child)) {
                .Pointer, .Array => asVec(i.child),
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

pub fn Child(comptime Vec: type) type {
    return switch(comptime @typeInfo(AsVec(Vec))) {
        .Vector => |i| i.child,
        .Array => |i| i.child,
        else => @compileError("Invalid type `" ++ @typeName(Vec) ++ "` passed to Child")
    };
}

pub fn vlen(comptime Vec: type) comptime_int {
    return switch(comptime @typeInfo(AsVec(Vec))) {
        .Array => |i| i.len,
        .Vector => |i| i.len,
        else => @compileError("Invalid type `" ++ @typeName(Vec) ++ "` passed to len")
    };
}

pub fn Mask(comptime Vec: type) type {
    return std.meta.Int(.unsigned, vlen(AsVec(Vec)));
}

pub fn UInt(comptime b: u16) type {
    return std.meta.Int(.unsigned, b);
}


pub fn splat(comptime length: comptime_int, comptime T: type, elem: T) @TypeOf(@splat(length, @as(T, elem))) {
    return @splat(length, @as(T, elem));
}

pub fn bits(comptime T: type) comptime_int {
    return @typeInfo(T).Int.bits; // XXX
}

pub fn WideInt(comptime by: comptime_int, comptime T: type) type {
    return std.meta.Int(@typeInfo(T).Int.signedness, @typeInfo(T).Int.bits * by);
}

pub fn NarrowInt(comptime by: comptime_int, comptime T: type) type {
    return std.meta.Int(@typeInfo(T).Int.signedness, bits(T) / by);
}

pub fn Resolve(comptime A: type, comptime B: type) type {
    return @TypeOf(@as(A, undefined), @as(B, undefined));
}

pub fn highInt(integer: anytype) NarrowInt(2, @TypeOf(integer)) {
    const T = @TypeOf(integer);
    return @intCast(NarrowInt(2, T), integer >> @divExact(bits(T), 2));
}

pub fn lowInt(integer: anytype) NarrowInt(2, @TypeOf(integer)) {
    const T = @TypeOf(integer);
    return @truncate(NarrowInt(2, T), integer);
}

pub fn high(v: anytype) T: {
    const T = AsVec(@TypeOf(v));
    const E = Child(T);
    break :T @Vector(vlen(T), NarrowInt(2, E));
} {
    const T = AsVec(@TypeOf(v));
    const E = Child(T);
    const R = @Vector(vlen(T), NarrowInt(2, E));

    return @intCast(R, asVec(v) >> splat(vlen(T), std.math.Log2Int(E), @divExact(bits(E), 2)));
}

pub fn low(v: anytype) T: {
    const T = AsVec(@TypeOf(v));
    const E = Child(T);
    break :T @Vector(vlen(T), NarrowInt(2, E));
} {
    const T = AsVec(@TypeOf(v));
    const E = Child(T);
    const R = @Vector(vlen(T), NarrowInt(2, E));

    return @intCast(R, asVec(v) & splat(vlen(T), E, std.math.maxInt(NarrowInt(2, E))));
}

pub fn Index(comptime len: u16) type {
    return std.math.Log2Int(std.meta.Int(.unsigned, len));
}

pub fn Count(comptime len: u16) type {
    return std.math.Log2Int(std.meta.Int(.unsigned, len + 1));
}
