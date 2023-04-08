const std = @import("std");
const simd = std.simd;
const meta = std.meta;

const util = @import("util.zig");
const vlen = util.len;
const Mask = util.Mask;
const Child = util.Child;
const asVec  = util.asVec;
const AsVec  = util.AsVec;
const splat = util.splat;
const WideInt = util.WideInt;
const NarrowInt = util.NarrowInt;
const high = util.high;
const low = util.low;
const Resolve = util.Resolve;
const UInt = util.UInt;

pub const BinaryOp = enum {
    Add,
    AddWrap,
    AddSat,

    Sub,
    SubWrap,
    SubSat,

    Mul,
    MulWrap,
    MulSat,

    Div,

    And,
    Or,
    Xor,

    Min,
    Max,

    fn perform(op: BinaryOp, a: anytype, b: anytype) @TypeOf(a, b) {
        return switch(op) {
            .Add => a + b,
            .AddWrap => a +% b,
            .AddSat => a +| b,

            .Sub => a - b,
            .SubWrap => a -% b,
            .SubSat => a -| b,

            .Mul => a * b,
            .MulWrap => a *% b,
            .MulSat => a *| b,

            .Div => a / b,

            .And => a & b,
            .Or => a | b,
            .Xor => a ^ b,

            .Min => @min(a, b),
            .Max => @max(a, b),
        };
    }
};

pub const UnaryOp = enum {
    Not,
    Square,

    fn perform(op: UnaryOp, a: anytype) @TypeOf(a) {
        return switch(op) {
            .Not => if(comptime @TypeOf(a) == bool) !a else ~a,
            .Square => a*a,
        };
    }

};

/// Adds every nb_elements elements in v together.
pub inline fn horizontalOp(
    comptime op: BinaryOp,
    comptime nb_elements: comptime_int,
    comptime len: u16,
    comptime E: type,
    v: @Vector(len, E),
) @Vector(@divExact(len, nb_elements), E) {

    const deinterlaced = simd.deinterlace(nb_elements, v);
    const D = @TypeOf(deinterlaced[0]);

    var accumulator: D = deinterlaced[0];

    inline for(deinterlaced[1..]) |e|
        accumulator = op.perform(accumulator, e);

    return accumulator;
}


/// Returns a mask of the high bits of each element in vector.
/// Related instructions: pmovmskb
pub inline fn movemask(comptime len: u16, comptime E: type, v: @Vector(len, E)) UInt(len) {
    if(comptime E != u8)
        @compileError("movemask not implemented for non-u8"); // XXX

    const hi_bits = splat(len, u8, 0x80);

    return @bitCast(UInt(len), v & hi_bits == hi_bits);
}

/// Multiplies the integer elements of a and b, doubling
/// their bit width, and horizontally adding pairs of adjacent
/// elements of the resultant vector.
/// Related instructions: pmaddwd
pub inline fn mulWithHorizontalAdd(
    comptime len: u16,
    comptime E: type,
    a: @Vector(len, E),
    b: @Vector(len, E),
) @Vector(@divExact(len, 2), WideInt(2, E)) {
    const W = WideInt(2, E);
    const D = @Vector(len, W);
    const a_wide = @as(D, a);
    const b_wide = @as(D, b);

    // Overflow can't happen
    const added = b: {@setRuntimeSafety(false); break :b a_wide * b_wide;};
    return horizontalOp(.AddWrap, 2, len, W, added);
}

/// Computes the absolute difference between each elements of a and b.
// XXX: optimize
pub fn absDiff(comptime len: u16, comptime E: type, a: @Vector(len, E), b: @Vector(len, E)) @TypeOf(a, b) {
    return @select(E, a > b, a -% b, b -% a);
}

/// Computes the sum of absolute differences between each elements of
/// a and b.
/// Related instructions: psadbw
// XXX: Make this generate psadbw
pub inline fn sad(comptime len: u16, comptime E: type, a: @Vector(len, E), b: @Vector(len, E)) WideInt(2, E) {
    const d = absDiff(len, E, a, b);

    // find a better way to do this?
    var accum: u16 = d[0];
    for(@as([len]E, d)[1..]) |e| accum += e;

    return accum;
}

/// Multiplies the integer elements of a and b, producing integers that are twice as wide.
pub inline fn mulWide(comptime len: u16, comptime E: type, a: @Vector(len, E), b: @Vector(len, E)) @Vector(len, WideInt(2, E)) {
    const W = @Vector(len, WideInt(2, E));

    return @as(W, a) * @as(W, b);
}

/// Multiplies the integer elements of a and b and returns a vector containing their high bits.
/// Related instructions: pmulhw
pub inline fn mulHigh(comptime len: u16, comptime E: type, a: @Vector(len, E), b: @Vector(len, E)) @Vector(len, E) {
    // XXX: inline asm
    return high(mulWide(len, E, a, b));
}

/// Multiplies the integer elements of a and b and returns a vector containing low bits.
/// Related instructions: pmullw
pub inline fn mulLow(comptime len: u16, comptime E: type, a: @Vector(len, E), b: @Vector(len, E)) @Vector(len, E) {
    // XXX: inline asm
    return low(mulWide(len, E, a, b));
}

/// Averages each of the elements of a and b.
/// Related instructions: pavgw
pub inline fn avg(comptime len: u16, comptime E: type, a: @Vector(len, E), b: @Vector(len, E)) @Vector(len, E) {
    // XXX: inline asm
    const one_E = splat(len, E, 1);
    const one_u1 = splat(len, u1, 1);
    return (a + b + one_E) >> one_u1;
}

/// Conditionally stores elements from src based on mask into dst.
/// Related instructions: maskmovq
pub inline fn maskmove(
    comptime len: u16,
    comptime E: type,
    dst: *[len]E,
    src: @Vector(len, E),
    mask: @Vector(len, bool)
) void {
    for(0..len) |i| {
        if(mask[i]) dst[i] = src[i];
    }
}

/// Alternate between subtracting and adding elements from a and b.
/// Related instructions: addsubpd
pub inline fn addsub(comptime len: u16, comptime E: type, a: @Vector(len, E), b: @Vector(len, E)) @Vector(len, E) {
    const de_a = simd.deinterlace(2, a);
    const de_b = simd.deinterlace(2, b);

    return simd.interlace(.{de_a[0] -% de_b[0], de_a[1] +% de_b[1]});
}

// Tests

test "horizontalOp" {
    const in = @Vector(8, u8){1, 2, 3, 4, 5, 6, 7, 8};
    const expected = @Vector(4, u8){3, 7, 11, 15};
    try std.testing.expectEqual(expected, horizontalOp(.Add, 2, 8, u8, in));
}

test "movemask" {
    const input = "amogus\xff\x80\x7f\x81".*;
    const expected: u10 = 0b1011000000;
    const actual = movemask(10, u8, input);
    try std.testing.expectEqual(expected, actual);
}

test "mulWithHorizontalAdd" {
    const a = @Vector(4, u8){255, 255, 3, 7};
    const b = @Vector(4, u8){255, 255, 42, 0};
    const expected  = @Vector(2, u16){64514, 126};
    const actual = mulWithHorizontalAdd(4, u8, a, b);
    try std.testing.expectEqual(expected, actual);
}

test "absDiff" {
    const a = @Vector(8, u8){255, 255, 3, 7, 4, 2, 0, 6};
    const b = @Vector(8, u8){255, 255, 9, 3, 1, 3, 3, 7};
    const expected  = @Vector(8, u8){0, 0, 6, 4, 3, 1, 3, 1};
    const actual = absDiff(8, u8, a, b);
    try std.testing.expectEqual(expected, actual);
}

test "sad" {
    const a = @Vector(8, u8){255, 255, 3, 7, 4, 2, 0, 6};
    const b = @Vector(8, u8){255, 255, 9, 3, 1, 3, 3, 7};
    const expected: u16 = 18;
    const actual = sad(8, u8, a, b);
    try std.testing.expectEqual(expected, actual);
}

test "mulWide" {
    const a = .{3, 1, 3, 3, 7};
    const b = .{7, 3, 3, 1, 3};
    const expected = @Vector(5, u6){21, 3, 9, 3, 21};
    const actual = mulWide(5, u3, a, b);
    try std.testing.expectEqual(expected, actual);
}

test "mulHigh" {
    const a = .{3, 1, 3, 3, 7};
    const b = .{7, 3, 3, 1, 3};
    const expected = @Vector(5, u4){1, 0, 0, 0, 1};
    const actual = mulHigh(5, u4, a, b);
    try std.testing.expectEqual(expected, actual);
}

test "mulLow" {
    const a = .{3, 1, 3, 3, 7};
    const b = .{7, 3, 3, 1, 3};
    const expected = @Vector(5, u8){5, 3, 9, 3, 5};
    const actual = mulLow(5, u4, a, b);
    try std.testing.expectEqual(expected, actual);
}

test "avg" {
    const a = .{3, 1, 3, 3, 7};
    const b = .{7, 3, 3, 1, 3};
    const expected = @Vector(5, u4){5, 2, 3, 2, 5};
    const actual = avg(5, u4, a, b);
    try std.testing.expectEqual(expected, actual);
}

test "maskmove" {
    const src = .{3, 1, 3, 3, 7};
    const mask = .{false, true, true, true, true};
    const expected = @Vector(5, u4){0, 1, 3, 3, 7};
    var actual = [_]u3{0, 0, 0, 0, 0};
    maskmove(5, u3, &actual, src, mask);
    try std.testing.expectEqual(expected, actual);
}

test "addsub" {
    const a = .{1, 3, 3, 7};
    const b = .{3, 3, 1, 3};
    const expected = @Vector(4, u3){6, 6, 2, 2};
    const actual = addsub(4, u3, a, b);
    try std.testing.expectEqual(expected, actual);
}
