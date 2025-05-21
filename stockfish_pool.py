# stockfish_pool.py

import chess
import chess.engine
import os
import stat
import concurrent.futures
import io
import chess.pgn
import logging
from typing import List, Dict, Any, Optional, Literal, TypedDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions
class EvaluationType(TypedDict):
    type: Literal["cp", "mate"]
    value: int

class TopMoveType(TypedDict):
    move: str
    uci: str
    centipawn: int
    is_best: bool
    move_line: List[str]

class PositionDataType(TypedDict):
    fen: str
    move_number: int
    ply: int
    move: Optional[str]
    color: Literal["w", "b"]
    evaluation: Optional[EvaluationType]
    top_moves: Optional[List[TopMoveType]]
    depth: Optional[int]
    eval_change: Optional[float]
    move_quality: Optional[Literal["best", "excellent", "good", "inaccuracy", "mistake", "blunder"]]
    position_type: Optional[str]

class KeyPositionType(TypedDict):
    fen: str
    move_number: int
    move: str
    color: Literal["w", "b"]
    eval_change: float
    move_quality: Literal["best", "excellent", "good", "inaccuracy", "mistake", "blunder"]
    description: str
    alternative_moves: List[Dict[str, Any]]
    position_type: str

class PlayerStatsType(TypedDict):
    move_count: int
    accuracy: float
    blunders: int
    mistakes: int
    inaccuracies: int
    good_moves: int
    excellent_moves: int
    best_moves: int
    avg_centipawn_loss: float

class AnalysisSummaryType(TypedDict):
    white: PlayerStatsType
    black: PlayerStatsType
    total_moves: int
    average_eval: float
    opening_moves: int
    middlegame_moves: int
    endgame_moves: int
    decisive_moves: List[int]

class StockfishAnalysisType(TypedDict):
    positions: List[PositionDataType]
    key_positions: List[KeyPositionType]
    summary: AnalysisSummaryType
    game_phases: Dict[str, List[int]]

class StockfishPool:
    """A pool for parallel Stockfish analysis using multiple processes."""

    def __init__(self, stockfish_path: str = None, num_processes: int = None, default_depth: int = 15):
        """Initialize the Stockfish pool with improved error handling and permissions management."""
        # Find Stockfish path if not provided
        if not stockfish_path:
            # Common locations for Linux
            possible_paths = [
                "./stockfish-ubuntu-x86-64-avx2",    # Local project binary
                "/usr/local/bin/stockfish",          # System-wide install
                "/usr/bin/stockfish",                # System-wide install
                "/usr/games/stockfish",              # Some Linux distributions
                "stockfish"                          # Search in PATH
            ]

            # Get the current working directory for logging
            current_directory = os.getcwd()
            logger.info(f"Current directory: {current_directory}")
            
            # List files in current directory for debugging
            try:
                files = os.listdir(current_directory)
                logger.info("Files in current directory: %s", files)
            except Exception as e:
                logger.error(f"Error listing directory: {e}")

            # Try each possible path
            for path in possible_paths:
                logger.info(f"Trying Stockfish path: {path}")
                if os.path.exists(path):
                    stockfish_path = path
                    logger.info(f"Found Stockfish at: {path}")
                    break
                elif os.system(f"which {path} > /dev/null 2>&1") == 0:
                    stockfish_path = path
                    logger.info(f"Found Stockfish in PATH: {path}")
                    break

        if not stockfish_path:
            raise ValueError("Stockfish executable not found. Please ensure it is installed and accessible.")

        # Verify binary exists and permissions
        self._verify_binary(stockfish_path)

        # Test binary execution
        try:
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            engine.quit()
            logger.info("Successfully tested Stockfish binary")
        except Exception as e:
            raise ValueError(f"Error testing Stockfish binary: {e}")

        # Determine number of processes
        if num_processes is None:
            num_processes = max(1, os.cpu_count() - 1)  # Leave one core free

        self.stockfish_path = stockfish_path
        self.num_processes = num_processes
        self.default_depth = default_depth
        logger.info(f"Initialized StockfishPool with {num_processes} processes")

    def _verify_binary(self, path: str) -> None:
        """Verify Stockfish binary exists and has correct permissions."""
        # Check if file exists
        if not os.path.exists(path):
            raise ValueError(f"Stockfish binary not found at {path}")

        # Check if it's a file (not a directory)
        if not os.path.isfile(path):
            raise ValueError(f"Path {path} is not a file")

        # Get current permissions
        try:
            current_mode = os.stat(path).st_mode
        except Exception as e:
            raise ValueError(f"Cannot access file permissions: {e}")

        # Check if file is executable
        if not current_mode & stat.S_IXUSR:
            logger.warning(f"Binary {path} is not executable, attempting to fix permissions")
            try:
                # Try adding execute for user
                os.chmod(path, current_mode | stat.S_IXUSR)
                new_mode = os.stat(path).st_mode
                if not new_mode & stat.S_IXUSR:
                    raise ValueError("Failed to set execute permission")
                logger.info("Successfully added execute permission")
            except Exception as e:
                raise ValueError(f"Cannot set execute permission: {e}. Try: chmod +x {path}")

        # Check if file is readable
        if not current_mode & stat.S_IRUSR:
            logger.warning(f"Binary {path} is not readable, attempting to fix permissions")
            try:
                # Try adding read for user
                os.chmod(path, current_mode | stat.S_IRUSR)
                new_mode = os.stat(path).st_mode
                if not new_mode & stat.S_IRUSR:
                    raise ValueError("Failed to set read permission")
                logger.info("Successfully added read permission")
            except Exception as e:
                raise ValueError(f"Cannot set read permission: {e}. Try: chmod +r {path}")

        # Check owner
        try:
            stat_info = os.stat(path)
            if stat_info.st_uid != os.getuid():
                logger.warning(f"Binary {path} is not owned by current user")
                # Just a warning, not a fatal error
        except Exception as e:
            logger.warning(f"Cannot check file ownership: {e}")

        logger.info(f"Binary {path} verified with correct permissions")

    def _verify_binary_access(self, path: str) -> bool:
        """Verify that we can actually execute the binary."""
        try:
            engine = chess.engine.SimpleEngine.popen_uci(path)
            engine.quit()
            logger.info(f"Successfully verified binary execution at {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to execute binary at {path}: {e}")
            return False

    def _try_fix_permissions(self, path: str) -> bool:
        """Try various methods to fix binary permissions."""
        try:
            current_mode = os.stat(path).st_mode
            
            # Try adding execute permission
            new_mode = current_mode | stat.S_IXUSR | stat.S_IRUSR
            os.chmod(path, new_mode)
            
            # Verify the changes
            if os.stat(path).st_mode & (stat.S_IXUSR | stat.S_IRUSR):
                logger.info(f"Successfully fixed permissions for {path}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to fix permissions: {e}")
            return False

    def evaluate_position(self, fen: str, depth: int = None) -> Dict[str, Any]:
        """Evaluate a single position with Stockfish."""
        if depth is None:
            depth = self.default_depth

        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            board = chess.Board(fen)
            position_type = self._classify_position_type(board)
            adjusted_depth = self._adjust_depth_for_position(board, depth, position_type)

            info = engine.analyse(
                board,
                chess.engine.Limit(depth=adjusted_depth),
                multipv=3
            )

            score = info[0]["score"].white()
            evaluation = {
                "type": "mate" if score.is_mate() else "cp",
                "value": score.mate() / 100 if score.is_mate() else score.score() / 100
            }

            top_moves = []
            for pv_info in info:
                if "pv" in pv_info and pv_info["pv"]:
                    moves_pv = pv_info["pv"]
                    move = moves_pv[0] if moves_pv else None
                    if move:
                        var_score = pv_info["score"].white()
                        var_value = 10000 if var_score.is_mate() and var_score.mate() > 0 else (
                            -10000 if var_score.is_mate() else var_score.score()
                        )
                        
                        san_move = board.san(move)
                        move_line = []
                        board_copy = board.copy(stack=False)
                        
                        for pv_move in moves_pv:
                            if pv_move in board_copy.legal_moves:
                                move_line.append(board_copy.san(pv_move))
                                board_copy.push(pv_move)
                            else:
                                break

                        top_moves.append({
                            "move": san_move,
                            "uci": move.uci(),
                            "centipawn": var_value / 100,
                            "is_best": False,
                            "move_line": move_line
                        })

            if top_moves:
                best_eval = max(move["centipawn"] for move in top_moves) if board.turn == chess.WHITE else \
                          min(move["centipawn"] for move in top_moves)
                
                for move in top_moves:
                    if move["centipawn"] == best_eval:
                        move["is_best"] = True

            return {
                "evaluation": evaluation,
                "top_moves": top_moves,
                "depth": adjusted_depth,
                "position_type": position_type
            }

        except Exception as e:
            logger.error(f"Error evaluating position {fen}: {e}")
            return {
                "evaluation": {"type": "error", "value": 0},
                "error": str(e)
            }
        finally:
            try:
                engine.quit()
            except Exception as e:
                logger.error(f"Error closing engine: {e}")

    def analyze_pgn(self, pgn_str: str, depth: int = None) -> StockfishAnalysisType:
        """Analyze a complete PGN."""
        if depth is None:
            depth = self.default_depth

        game = chess.pgn.read_game(io.StringIO(pgn_str))
        if not game:
            return {"error": "Invalid PGN"}

        positions = []
        board = game.board()
        ply = 0
        nodes = list(game.mainline())

        for node in nodes:
            move = node.move
            san_move = board.san(move)
            current_color = "w" if ply % 2 == 0 else "b"
            current_move_number = (ply // 2) + 1

            board.push(move)
            
            position = {
                "fen": board.fen(),
                "move_number": current_move_number,
                "ply": ply,
                "move": san_move,
                "is_initial": False,
                "color": current_color,
            }
            positions.append(position)
            ply += 1

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            future_to_position = {
                executor.submit(self.evaluate_position, position["fen"], depth): i
                for i, position in enumerate(positions)
            }

            analyzed_positions = positions.copy()
            for future in concurrent.futures.as_completed(future_to_position):
                position_index = future_to_position[future]
                try:
                    analysis_result = future.result()
                    analyzed_positions[position_index].update(analysis_result)
                except Exception as e:
                    logger.error(f"Error analyzing position {position_index}: {e}")

        converted_positions = self._convert_positions_format(analyzed_positions)
        self._process_analysis_results(converted_positions)
        game_phases = self._identify_game_phases(converted_positions)

        return {
            "positions": converted_positions,
            "key_positions": self._identify_key_positions(converted_positions),
            "summary": self._generate_analysis_summary(converted_positions),
            "game_phases": game_phases
        }

    def _classify_position_type(self, board: chess.Board) -> str:
        """Classify the position type based on board state."""
        total_material = self._count_material(board, chess.WHITE) + self._count_material(board, chess.BLACK)
        piece_count = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK)) + \
                     len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) + \
                     len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                     len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)) + \
                     len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))

        if total_material <= 20 or piece_count <= 10:
            return "endgame"
        elif board.fullmove_number <= 10:
            return "opening"
        else:
            return "middlegame"

    def _count_material(self, board: chess.Board, color: chess.Color) -> int:
        """Count material value for a side."""
        material = len(board.pieces(chess.PAWN, color)) * 1
        material += len(board.pieces(chess.KNIGHT, color)) * 3
        material += len(board.pieces(chess.BISHOP, color)) * 3
        material += len(board.pieces(chess.ROOK, color)) * 5
        material += len(board.pieces(chess.QUEEN, color)) * 9
        return material

    def _adjust_depth_for_position(self, board: chess.Board, base_depth: int, position_type: str) -> int:
        """Adjust analysis depth based on position complexity."""
        adjusted_depth = base_depth

        if position_type == "tactical":
            adjusted_depth = base_depth + 3  # Deeper analysis for tactical positions
        elif position_type == "endgame" and self._count_material(board, chess.WHITE) + self._count_material(board, chess.BLACK) <= 15:
            adjusted_depth = base_depth + 3  # Deeper for simple endgames
        elif position_type == "opening" and board.fullmove_number <= 5:
            adjusted_depth = base_depth - 3  # Less depth needed for early opening

        return min(adjusted_depth, 18)  # Cap at maximum depth of 18

    def _process_analysis_results(self, positions: List[Dict[str, Any]]) -> None:
        """Process analysis results to add additional metrics."""
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]

            if "evaluation" not in prev_pos or "evaluation" not in curr_pos:
                continue

            prev_eval = prev_pos["evaluation"]
            curr_eval = curr_pos["evaluation"]

            if prev_eval.get("type") == "error" or curr_eval.get("type") == "error":
                continue

            prev_value = prev_eval["value"]
            curr_value = curr_eval["value"]

            # Convert mate scores to centipawns
            if prev_eval["type"] == "mate":
                prev_value = 10000 if prev_value > 0 else -10000
            if curr_eval["type"] == "mate":
                curr_value = 10000 if curr_value > 0 else -10000

            raw_eval_change = curr_value - prev_value
            curr_pos["raw_eval_change"] = raw_eval_change

            # Calculate change from player's perspective
            if curr_pos["color"] == "w":
                player_eval_change = raw_eval_change
            else:
                player_eval_change = -raw_eval_change

            curr_pos["eval_change"] = player_eval_change

    def _classify_move_quality(self, played_move_eval: Optional[float] = None,
                             best_move_eval: Optional[float] = None,
                             is_best_move: bool = False,
                             eval_change: Optional[float] = None) -> str:
        """Classify move quality based on evaluation."""
        if is_best_move:
            if eval_change is not None and eval_change < -100:
                return "forced"
            return "best"

        if played_move_eval is not None and best_move_eval is not None:
            eval_diff = played_move_eval - best_move_eval

            if eval_diff >= -10:  # Within 0.1 pawns
                return "excellent"
            elif eval_diff >= -30:  # Within 0.3 pawns
                return "good"
            elif eval_diff >= -70:  # Within 0.7 pawns
                return "inaccuracy"
            elif eval_diff >= -150:  # Within 1.5 pawns
                return "mistake"
            else:
                return "blunder"

        if eval_change is not None:
            if eval_change >= 0:
                return "good"
            elif eval_change > -50:
                return "inaccuracy"
            elif eval_change > -150:
                return "mistake"
            else:
                return "blunder"

        return "normal"

    def _generate_analysis_summary(self, positions: List[PositionDataType]) -> AnalysisSummaryType:
        """Generate a summary of the analysis."""
        valid_positions = [pos for pos in positions if pos.get("move")]
        white_moves = [pos for pos in valid_positions if pos.get("color") == "w"]
        black_moves = [pos for pos in valid_positions if pos.get("color") == "b"]

        white_stats = self._calculate_player_stats(white_moves)
        black_stats = self._calculate_player_stats(black_moves)

        # Calculate average evaluation
        evals = [pos.get("evaluation", {}).get("value", 0)
                for pos in valid_positions if "evaluation" in pos]
        avg_eval = sum(evals) / max(1, len(evals))

        opening_moves = sum(1 for pos in valid_positions if pos.get("position_type") == "opening")
        middlegame_moves = sum(1 for pos in valid_positions if pos.get("position_type") == "middlegame")
        endgame_moves = sum(1 for pos in valid_positions if pos.get("position_type") == "endgame")
        decisive_moves = [pos["move_number"] for pos in valid_positions
                        if "eval_change" in pos and abs(pos["eval_change"]) > 200]

        return {
            "white": white_stats,
            "black": black_stats,
            "total_moves": len(valid_positions),
            "average_eval": avg_eval,
            "opening_moves": opening_moves,
            "middlegame_moves": middlegame_moves,
            "endgame_moves": endgame_moves,
            "decisive_moves": decisive_moves
        }

    def _calculate_player_stats(self, moves: List[PositionDataType]) -> PlayerStatsType:
        """Calculate statistics for a player."""
        if not moves:
            return {
                "move_count": 0,
                "accuracy": 0,
                "blunders": 0,
                "mistakes": 0,
                "inaccuracies": 0,
                "good_moves": 0,
                "excellent_moves": 0,
                "best_moves": 0,
                "avg_centipawn_loss": 0
            }

        blunders = sum(1 for move in moves if move.get("move_quality") == "blunder")
        mistakes = sum(1 for move in moves if move.get("move_quality") == "mistake")
        inaccuracies = sum(1 for move in moves if move.get("move_quality") == "inaccuracy")
        good_moves = sum(1 for move in moves if move.get("move_quality") == "good")
        excellent_moves = sum(1 for move in moves if move.get("move_quality") == "excellent")
        best_moves = sum(1 for move in moves if move.get("move_quality") == "best")

        centipawn_losses = [abs(move.get("eval_change", 0)) for move in moves
                          if "eval_change" in move and move.get("eval_change", 0) < 0]
        avg_centipawn_loss = sum(centipawn_losses) / max(1, len(centipawn_losses))
        accuracy = max(0, min(100, 100 - min(100, avg_centipawn_loss / 10)))

        return {
            "move_count": len(moves),
            "accuracy": round(accuracy, 1),
            "blunders": blunders,
            "mistakes": mistakes,
            "inaccuracies": inaccuracies,
            "good_moves": good_moves,
            "excellent_moves": excellent_moves,
            "best_moves": best_moves,
            "avg_centipawn_loss": round(avg_centipawn_loss, 1)
        }
