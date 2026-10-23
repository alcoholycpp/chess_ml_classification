#!/usr/bin/env python3
import csv, io, random, sys
import chess, chess.pgn
import numpy as np, pandas as pd
from tqdm import tqdm


PGN = sys.argv[1] if len(sys.argv) > 1 else "lichess_db_standard_rated_2015-11.pgn"
OUT = "positions.csv"
MAX_GAMES = 200_000          # analysed games to use
PER_GAME = 5                 # positions sampled per game
MIN_MOVE = 8                 # skip the first 8 full moves (opening theory)
CLIP = 10.0                  # eval and mate scores clipped to +-10 pawns
RESULT = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}
PIECES = {"pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
          "rook": chess.ROOK, "queen": chess.QUEEN}
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]
W, B = chess.WHITE, chess.BLACK
random.seed(42)
 
 
def raw_games(path):
    buf = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("[Event ") and buf:
                yield "".join(buf)
                buf = []
            buf.append(line)
    if buf:
        yield "".join(buf)
 
 
def eval_pawns(node):
    e = node.eval()
    if e is None:
        return None
    return max(-CLIP, min(CLIP, e.white().score(mate_score=1000) / 100))
 
 
def features(b):
    f = {f"{n}_diff": len(b.pieces(t, W)) - len(b.pieces(t, B)) for n, t in PIECES.items()}
    f["n_pieces"] = len(b.piece_map())
 
    mine = b.legal_moves.count()
    b.push(chess.Move.null())
    theirs = b.legal_moves.count()
    b.pop()
    f["mobility_diff"] = mine - theirs if b.turn == W else theirs - mine
 
    f["center_diff"] = sum(b.is_attacked_by(W, s) - b.is_attacked_by(B, s) for s in CENTER)
 
    def exposure(c):    # squares around c's king attacked by the enemy
        return sum(b.is_attacked_by(not c, s) for s in b.attacks(b.king(c)))
 
    def doubled(c):
        files = [chess.square_file(s) for s in b.pieces(chess.PAWN, c)]
        return len(files) - len(set(files))
 
    f["king_exposure_diff"] = exposure(B) - exposure(W)
    f["doubled_pawns_diff"] = doubled(B) - doubled(W)
    f["is_check"] = int(b.is_check())
    f["white_to_move"] = int(b.turn == W)
    f["move"] = b.fullmove_number
    return f
 
 
seen = used = 0
writer = None
bar = tqdm(total=MAX_GAMES, desc="analysed games")
with open(OUT, "w", newline="") as out:
    for text in raw_games(PGN):
        seen += 1
        if "[%eval" not in text:
            continue
        game = chess.pgn.read_game(io.StringIO(text))
        result = RESULT.get(game.headers.get("Result"))
        nodes = list(game.mainline())
        ok = [i for i, n in enumerate(nodes) if i >= 2 * MIN_MOVE and n.eval() is not None]
        if result is None or len(ok) < PER_GAME:
            continue
 
        pick = set(random.sample(ok, PER_GAME))
        gid = game.headers["Site"].rsplit("/", 1)[-1]
        board = game.board()
        for i, n in enumerate(nodes):
            board.push(n.move)
            if i in pick:
                row = {"game_id": gid, **features(board),
                       "sf_eval": eval_pawns(n), "result": result}
                if writer is None:
                    writer = csv.DictWriter(out, row.keys())
                    writer.writeheader()
                writer.writerow(row)
            if i >= max(pick):
                break
 
        used += 1
        bar.update()
        if used == MAX_GAMES:
            break
bar.close()
 
df = pd.read_csv(OUT)
label = np.select([df.sf_eval > 1, df.sf_eval < -1], ["W", "B"], "D")
print(f"scanned {seen:,} games, used {used:,}, wrote {len(df):,} positions to {OUT}")
print("\nlabel from sf_eval (|eval| <= 1 -> D):")
print(pd.Series(label).value_counts(normalize=True).round(3))
print("\ngame result (1 = White won, 0 = draw, -1 = Black won):")
print(df.result.value_counts(normalize=True).round(3))
 
