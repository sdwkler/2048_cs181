import sys
import math
import random
import struct
import typing
import abc

def info(*argv) -> None:
    print(*argv, file=sys.stdout)

def error(*argv) -> None:
    print(*argv, file=sys.stderr)

def debug(*argv) -> None:
    print(*argv, file=sys.stderr)

class board:
    """
    64-bit bitboard implementation for 2048

    index:
     0  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15

    note that the 64-bit raw value is stored in little endian
    i.e., 0x4312752186532731 is displayed as
    +------------------------+
    |     2     8   128     4|
    |     8    32    64   256|
    |     2     4    32   128|
    |     4     2     8    16|
    +------------------------+
    """

    def __init__(self, raw : int = 0):
        self.raw = int(raw)

    def __int__(self) -> int:
        return self.raw

    def fetch(self, i : int) -> int:
        """
        get a 16-bit row
        """
        return (self.raw >> (i << 4)) & 0xffff

    def place(self, i : int, r : int) -> None:
        """
        set a 16-bit row
        """
        self.raw = (self.raw & ~(0xffff << (i << 4))) | ((r & 0xffff) << (i << 4))

    def at(self, i : int) -> int:
        """
        get a 4-bit tile
        """
        return (self.raw >> (i << 2)) & 0x0f

    def set(self, i : int, t : int) -> None:
        """
        set a 4-bit tile
        """
        self.raw = (self.raw & ~(0x0f << (i << 2))) | ((t & 0x0f) << (i << 2))

    def __getitem__(self, i : int) -> int:
        return self.at(i)

    def __setitem__(self, i : int, t : int) -> None:
        self.set(i, t)

    def __eq__(self, other) -> bool:
        return isinstance(other, board) and self.raw == other.raw

    def __lt__(self, other) -> bool:
        return isinstance(other, board) and self.raw < other.raw

    def __ne__(self, other) -> bool:
        return not self == other

    def __gt__(self, other) -> bool:
        return isinstance(other, board) and other < self

    def __le__(self, other) -> bool:
        return isinstance(other, board) and not other < self

    def __ge__(self, other) -> bool:
        return isinstance(other, board) and not self < other

    class lookup:
        """
        the lookup table for sliding board
        """

        find = [None] * 65536

        class entry:
            def __init__(self, row : int):
                V = [ (row >> 0) & 0x0f, (row >> 4) & 0x0f, (row >> 8) & 0x0f, (row >> 12) & 0x0f ]
                L, score = board.lookup.entry.mvleft(V)
                V.reverse() # mirror
                R, score = board.lookup.entry.mvleft(V)
                R.reverse()
                self.raw = row # base row (16-bit raw)
                self.left = (L[0] << 0) | (L[1] << 4) | (L[2] << 8) | (L[3] << 12) # left operation
                self.right = (R[0] << 0) | (R[1] << 4) | (R[2] << 8) | (R[3] << 12) # right operation
                self.score = score # merge reward

            def move_left(self, raw : int, sc : int, i : int) -> tuple[int, int]:
                return raw | (self.left << (i << 4)), sc + self.score

            def move_right(self, raw : int, sc : int, i : int) -> tuple[int, int]:
                return raw | (self.right << (i << 4)), sc + self.score

            @staticmethod
            def mvleft(row : int) -> tuple[list[int], int]:
                buf = [t for t in row if t]
                res, score = [], 0
                while buf:
                    if len(buf) >= 2 and buf[0] is buf[1]:
                        buf = buf[1:]
                        buf[0] += 1
                        score += 1 << buf[0]
                    res += [buf[0]]
                    buf = buf[1:]
                return res + [0] * (4 - len(res)), score

        @classmethod
        def init(cls) -> None:
            cls.find = [cls.entry(row) for row in range(65536)]

    def init(self) -> None:
        """
        reset to initial state, i.e., witn only 2 random tiles on board
        """
        self.raw = 0
        self.popup()
        self.popup()

    def popup(self) -> None:
        """
        add a new random tile on board, or do nothing if the board is full
        2-tile: 90%
        4-tile: 10%
        """
        space = [i for i in range(16) if self.at(i) == 0]
        if space:
            self.set(random.choice(space), 1 if random.random() < 0.9 else 2)

    def move(self, opcode : int) -> int:
        """
        apply an action to the board
        return the reward of the action, or -1 if the action is illegal
        """
        if opcode == 0:
            return self.move_up()
        elif opcode == 1:
            return self.move_right()
        elif opcode == 2:
            return self.move_down()
        elif opcode == 3:
            return self.move_left()
        else:
            return -1

    def move_left(self) -> int:
        move = 0
        prev = self.raw
        score = 0
        for i in range(4):
            move, score = self.lookup.find[self.fetch(i)].move_left(move, score, i)
        self.raw = move
        return score if move != prev else -1

    def move_right(self) -> int:
        move = 0
        prev = self.raw
        score = 0
        for i in range(4):
            move, score = self.lookup.find[self.fetch(i)].move_right(move, score, i)
        self.raw = move
        return score if move != prev else -1

    def move_up(self) -> int:
        self.rotate_clockwise()
        score = self.move_right()
        self.rotate_counterclockwise()
        return score

    def move_down(self) -> int:
        self.rotate_clockwise()
        score = self.move_left()
        self.rotate_counterclockwise()
        return score

    def transpose(self) -> None:
        """
        swap rows and columns
        +------------------------+       +------------------------+
        |     2     8   128     4|       |     2     8     2     4|
        |     8    32    64   256|       |     8    32     4     2|
        |     2     4    32   128| ----> |   128    64    32     8|
        |     4     2     8    16|       |     4   256   128    16|
        +------------------------+       +------------------------+
        """
        self.raw = (self.raw & 0xf0f00f0ff0f00f0f) | ((self.raw & 0x0000f0f00000f0f0) << 12) | ((self.raw & 0x0f0f00000f0f0000) >> 12)
        self.raw = (self.raw & 0xff00ff0000ff00ff) | ((self.raw & 0x00000000ff00ff00) << 24) | ((self.raw & 0x00ff00ff00000000) >> 24)

    def mirror(self) -> None:
        """
        reflect the board horizontally, i.e., exchange columns
        +------------------------+       +------------------------+
        |     2     8   128     4|       |     4   128     8     2|
        |     8    32    64   256|       |   256    64    32     8|
        |     2     4    32   128| ----> |   128    32     4     2|
        |     4     2     8    16|       |    16     8     2     4|
        +------------------------+       +------------------------+
        """
        self.raw = ((self.raw & 0x000f000f000f000f) << 12) | ((self.raw & 0x00f000f000f000f0) << 4) \
                 | ((self.raw & 0x0f000f000f000f00) >> 4) | ((self.raw & 0xf000f000f000f000) >> 12)

    def flip(self) -> None:
        """
        reflect the board vertically, i.e., exchange rows
        +------------------------+       +------------------------+
        |     2     8   128     4|       |     4     2     8    16|
        |     8    32    64   256|       |     2     4    32   128|
        |     2     4    32   128| ----> |     8    32    64   256|
        |     4     2     8    16|       |     2     8   128     4|
        +------------------------+       +------------------------+
        """
        self.raw = ((self.raw & 0x000000000000ffff) << 48) | ((self.raw & 0x00000000ffff0000) << 16) \
                 | ((self.raw & 0x0000ffff00000000) >> 16) | ((self.raw & 0xffff000000000000) >> 48)

    def rotate(self, r : int = 1) -> None:
        """
        rotate the board clockwise by given times
        """
        r = ((r % 4) + 4) % 4
        if r == 0:
            pass
        elif r == 1:
            self.rotate_clockwise()
        elif r == 2:
            self.reverse()
        elif r == 3:
            self.rotate_counterclockwise()

    def rotate_clockwise(self) -> None:
        self.transpose()
        self.mirror()

    def rotate_counterclockwise(self) -> None:
        self.transpose()
        self.flip()

    def reverse(self) -> None:
        self.mirror()
        self.flip()

    def __str__(self) -> str:
        state = '+' + '-' * 24 + '+\n'
        for i in range(0, 16, 4):
            state += ('|' + ''.join('{0:6d}'.format((1 << self.at(j)) & -2) for j in range(i, i + 4)) + '|\n')
            # use -2 (0xff...fe) to remove the unnecessary 1 for (1 << 0)
        state += '+' + '-' * 24 + '+'
        return state