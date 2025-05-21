# stockfish_pool.py

import chess
import chess.engine
import os
import concurrent.futures
import io
import chess.pgn
from typing import List, Dict, Any, Optional, Literal, TypedDict

# Type definitions for Stockfish analysis
class EvaluationType(TypedDict):
    type: Literal["cp", "mate"]
    value: int

class TopMoveType(TypedDict):
    move: str
    uci: str
    centipawn: int
    is_best: bool
    move_line: List[str]  # List of moves in the line (SAN format)

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
    position_type: Optional[str]  # New: classify position type (e.g., tactical, strategic, endgame)

class KeyPositionType(TypedDict):
    fen: str
    move_number: int
    move: str
    color: Literal["w", "b"]
    eval_change: float
    move_quality: Literal["best", "excellent", "good", "inaccuracy", "mistake", "blunder"]
    description: str
    alternative_moves: List[Dict[str, Any]]  # New: include alternative moves
    position_type: str  # New: position classification

class PlayerStatsType(TypedDict):
    move_count: int
    accuracy: float
    blunders: int
    mistakes: int
    inaccuracies: int
    good_moves: int
    excellent_moves: int
    best_moves: int
    avg_centipawn_loss: float  # New: average centipawn loss

class AnalysisSummaryType(TypedDict):
    white: PlayerStatsType
    black: PlayerStatsType
    total_moves: int
    average_eval: float
    opening_moves: int  # New: number of opening moves
    middlegame_moves: int  # New: number of middlegame moves
    endgame_moves: int  # New: number of endgame moves
    decisive_moves: List[int]  # New: list of decisive move numbers

class StockfishAnalysisType(TypedDict):
    positions: List[PositionDataType]
    key_positions: List[KeyPositionType]
    summary: AnalysisSummaryType
    game_phases: Dict[str, List[int]]  # New: mapping of game phases to move numbers

class StockfishPool:
    """
    An enhanced pool for parallel Stockfish analysis using multiple processes.
    """

    def __init__(self, stockfish_path: str = None, num_processes: int = None, default_depth: int = 15):
        """
        Initialize the Stockfish pool.

        Args:
            stockfish_path: Path to Stockfish executable
            num_processes: Number of processes to use (defaults to CPU count - 1)
        """
        # Find Stockfish path if not provided
        if not stockfish_path:
            # Common locations
            possible_paths = [
                "/usr/local/bin/stockfish",
                "/usr/bin/stockfish",
                "stockfish"  # Rely on PATH
            ]

            for path in possible_paths:
                if os.path.exists(path) or os.system(f"which {path} > /dev/null 2>&1") == 0:
                    stockfish_path = path
                    break

        if not stockfish_path:
            raise ValueError("Stockfish executable not found. Please provide the path.")

        # Determine number of processes
        if num_processes is None:
            num_processes = max(1, os.cpu_count() - 1)  # Leave one core free

        self.stockfish_path = stockfish_path
        self.num_processes = num_processes
        self.default_depth = default_depth

    def evaluate_position(self, fen: str, depth: int = None) -> Dict[str, Any]:
        """
        Evaluate a single position with Stockfish with enhanced analysis.
        Improved implementation that analyzes the position directly from FEN.

        Args:
            fen: FEN string representing the position
            depth: Analysis depth (uses default_depth if None)

        Returns:
            Position data with evaluation and top moves
        """
        if depth is None:
            depth = self.default_depth

        # Create a new engine instance for this process
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

        try:
            # Set up the position from FEN
            board = chess.Board(fen)
            # Determine position type
            position_type = self._classify_position_type(board)

            # Adjust depth based on position complexity
            adjusted_depth = self._adjust_depth_for_position(board, depth, position_type)

            info = engine.analyse(
                board,
                chess.engine.Limit(depth=adjusted_depth),
                multipv=3  # Analyze top 3 moves (reduced from 5 for better performance)
            )

            score = info[0]["score"].white()

            if score.is_mate():
                evaluation = {
                    "type": "mate",
                    "value": score.mate() / 100
                }
            else:
                evaluation = {
                    "type": "cp",
                    "value": score.score() / 100
                }
            # Get top moves from this position
            top_moves = []
            # Run multi-variation analysis
            for pv_info in info:
                # Process all moves returned by the engine (should be 3 with multipv=3)
                if "pv" in pv_info and pv_info["pv"]:
                    moves_pv = pv_info["pv"]
                    move = moves_pv[0] if moves_pv else None
                    if move:
                        # Get score for this variation (from White's perspective)
                        var_score = pv_info["score"].white()

                        # Convert mate scores to centipawns
                        if var_score.is_mate():
                            var_value = 10000 if var_score.mate() > 0 else -10000
                        else:
                            var_value = var_score.score()

                        # Convert move to algebraic notation
                        san_move = board.san(move)

                        # Generate the full move line in SAN notation
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
                            "is_best": False,  # Will set this after collecting all moves
                            "move_line": move_line
                        })

            # Determine the best move based on evaluation
            if top_moves:
                # For white's turn, higher evaluation is better
                # For black's turn, lower evaluation is better
                if board.turn == chess.WHITE:
                    best_eval = max(move["centipawn"] for move in top_moves)
                else:
                    best_eval = min(move["centipawn"] for move in top_moves)

                # Mark the best move(s)
                for move in top_moves:
                    if move["centipawn"] == best_eval:
                        move["is_best"] = True

            # Return analysis results
            return {
                "evaluation": evaluation,
                "top_moves": top_moves,
                "depth": adjusted_depth,
                "position_type": position_type
            }

        except Exception as e:
            print(f"Error evaluating position {fen}: {e}")
            return {
                "evaluation": {"type": "error", "value": 0},
                "error": str(e)
            }

        finally:
            # Always close the engine
            engine.quit()

    def _classify_position_type(self, board: chess.Board) -> str:
        """
        Classify the position type based on board state.

        Args:
            board: Chess board

        Returns:
            Position type classification
        """
        # Count material
        white_material = self._count_material(board, chess.WHITE)
        black_material = self._count_material(board, chess.BLACK)
        total_material = white_material + black_material

        # Count pieces
        piece_count = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK)) + \
                     len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) + \
                     len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                     len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)) + \
                     len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))

        # Check for endgame
        if total_material <= 20 or piece_count <= 10:
            return "endgame"

        # Check for opening
        if board.fullmove_number <= 10:
            return "opening"

        # Check for tactical position
        # if self._is_tactical_position(board):
        #     return "tactical"

        # Default to middlegame
        return "middlegame"

    def _count_material(self, board: chess.Board, color: chess.Color) -> int:
        """
        Count material value for a side.

        Args:
            board: Chess board
            color: Color to count material for

        Returns:
            Material value
        """
        material = 0
        material += len(board.pieces(chess.PAWN, color)) * 1
        material += len(board.pieces(chess.KNIGHT, color)) * 3
        material += len(board.pieces(chess.BISHOP, color)) * 3
        material += len(board.pieces(chess.ROOK, color)) * 5
        material += len(board.pieces(chess.QUEEN, color)) * 9
        return material

    def _is_tactical_position(self, board: chess.Board) -> bool:
        """
        Determine if a position is tactical.

        Args:
            board: Chess board

        Returns:
            True if position is tactical
        """
        # Check for checks
        if board.is_check():
            return True

        # Check for captures
        for move in board.legal_moves:
            if board.is_capture(move):
                return True

        return False

    def _adjust_depth_for_position(self, board: chess.Board, base_depth: int, position_type: str) -> int:
        """
        Adjust analysis depth based on position complexity.
        Uses more moderate depth adjustments to improve performance.

        Args:
            board: Chess board
            base_depth: Base depth
            position_type: Position type

        Returns:
            Adjusted depth (capped at 18 for complex positions)
        """
        adjusted_depth = base_depth

        # Adjust based on position type
        if position_type == "tactical":
            adjusted_depth = base_depth + 3  # Deeper analysis for tactical positions
        elif position_type == "endgame" and self._count_material(board, chess.WHITE) + self._count_material(board, chess.BLACK) <= 15:
            adjusted_depth = base_depth + 3  # Deeper for simple endgames
        elif position_type == "opening" and board.fullmove_number <= 5:
            adjusted_depth = base_depth - 3  # Less depth needed for early opening

        # Cap at maximum depth of 18
        return min(adjusted_depth, 18)

    def analyze_pgn(self, pgn_str: str, depth: int = None) -> StockfishAnalysisType:
        """
        Analyze a complete PGN using parallel processing with ProcessPoolExecutor.
        Improved implementation that correctly associates moves with evaluations.

        Args:
            pgn_str: PGN string to analyze
            depth: Analysis depth (uses default_depth if None)

        Returns:
            Dictionary with analysis results
        """
        if depth is None:
            depth = self.default_depth

        # Parse PGN
        game = chess.pgn.read_game(io.StringIO(pgn_str))
        if not game:
            return {"error": "Invalid PGN"}

        # Extract positions and moves
        positions = []
        board = game.board()

        # Process moves - we'll create a position for each move in the game
        ply = 0
        nodes = list(game.mainline())

        for i, node in enumerate(nodes):
            move = node.move
            san_move = board.san(move)

            # Determine whose move it is
            current_color = "w" if ply % 2 == 0 else "b"
            current_move_number = (ply // 2) + 1  # 1-based move numbering

            # This position will store the move that was played from it
            board.push(move)
            
            position = {
                "fen": board.fen(),
                "move_number": current_move_number,
                "ply": ply,
                "move": san_move,  # The move that was actually played to result in the fen
                "is_initial": False,
                "color": current_color,  # Who moves next from this position
            }

            # Store the position
            positions.append(position)

            ply += 1
            

        # Use ProcessPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit all positions for analysis
            future_to_position = {
                executor.submit(self.evaluate_position, position["fen"], depth): i
                for i, position in enumerate(positions)
            }

            # Process results as they complete
            analyzed_positions = positions.copy()  # Start with original positions
    
            for future in concurrent.futures.as_completed(future_to_position):
                position_index = future_to_position[future]
                try:
                    # Get analysis results
                    analysis_result = future.result()

                    # Update position with analysis results
                    analyzed_positions[position_index].update(analysis_result)

                except Exception as e:
                    print(f"Error analyzing position {position_index}: {e}")
                    # Keep original position data

        # # Convert positions to the format expected by the rest of the code
        # converted_positions = self._convert_positions_format(analyzed_positions)

        # # Process results to add additional metrics
        # self._process_analysis_results(converted_positions)

        # # Identify game phases
        # game_phases = self._identify_game_phases(converted_positions)

        # return {
        #     "positions": converted_positions,
        #     "key_positions": self._identify_key_positions(converted_positions),
        #     "summary": self._generate_analysis_summary(converted_positions),
        #     "game_phases": game_phases
        # }

        return analyzed_positions



    def _convert_positions_format(self, analyzed_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert the new position format to the format expected by the rest of the code.
        This ensures backward compatibility with existing functions.

        Args:
            analyzed_positions: List of analyzed positions in the new format

        Returns:
            List of positions in the format expected by the rest of the code
        """
        converted_positions = []

        for pos in analyzed_positions:
            # Skip the initial position
            if pos.get("is_initial", False):
                continue

            # Create a position in the old format
            converted_pos = {
                "fen": pos.get("resulting_fen", pos.get("fen", "")),  # Use resulting FEN if available
                "move_number": pos.get("move_number", 0),
                "ply": pos.get("ply", 0),
                "move": pos.get("move", ""),  # Use the move that was played
                "color": pos.get("color", "w"),  # Color that made the move
                "position_type": pos.get("position_type", "middlegame"),
                "evaluation": pos.get("evaluation", {"type": "cp", "value": 0})
            }

            # Add top moves - these are the moves that could have been played
            if "top_moves" in pos:
                converted_pos["top_moves"] = []
                for top_move in pos["top_moves"]:
                    # Check if this was the move that was actually played
                    is_played = top_move["move"] == pos.get("move", "")

                    converted_pos["top_moves"].append({
                        "move": top_move["move"],
                        "centipawn": top_move.get("centipawn", 0),
                        "is_best": top_move.get("is_best", False),
                        "is_played": is_played,
                        "move_line": top_move.get("move_line", [])
                    })

            converted_positions.append(converted_pos)

        return converted_positions

    def _process_analysis_results(self, positions: List[Dict[str, Any]]) -> None:
        """
        Process analysis results to add additional metrics.
        Improved implementation with clearer perspective handling.

        Args:
            positions: List of analyzed positions
        """
        # Skip if no positions
        if not positions:
            return

        # Calculate evaluation changes
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]

            # Skip if missing evaluations
            if "evaluation" not in prev_pos or "evaluation" not in curr_pos:
                continue

            prev_eval = prev_pos["evaluation"]
            curr_eval = curr_pos["evaluation"]

            # Skip if error in evaluations
            if prev_eval.get("type") == "error" or curr_eval.get("type") == "error":
                continue

            # Convert mate scores to centipawns
            prev_value = prev_eval["value"]
            if prev_eval["type"] == "mate":
                prev_value = 10000 if prev_value > 0 else -10000

            curr_value = curr_eval["value"]
            if curr_eval["type"] == "mate":
                curr_value = 10000 if curr_value > 0 else -10000

            # Calculate raw evaluation change (from White's perspective)
            raw_eval_change = curr_value - prev_value

            # Calculate change from perspective of player who made the move
            if curr_pos["color"] == "w":  # White moved
                # For White, positive change is good (improving White's position)
                player_eval_change = raw_eval_change
            else:  # Black moved
                # For Black, negative change is good (decreasing White's advantage)
                player_eval_change = -raw_eval_change

            # Store both perspectives
            curr_pos["raw_eval_change"] = raw_eval_change  # White's perspective
            curr_pos["eval_change"] = player_eval_change  # Player's perspective

            # Get information needed for move quality classification
            # Find the played move and best move evaluations
            played_move_eval = None
            best_move_eval = None
            is_best_move = False

            if "top_moves" in curr_pos:
                # Find the played move in top moves
                played_move = curr_pos.get("move")
                for move in curr_pos.get("top_moves", []):
                    if move.get("move") == played_move:
                        played_move_eval = move.get("centipawn")
                        is_best_move = move.get("is_best", False)
                        break

                # Find the best move evaluation
                for move in curr_pos.get("top_moves", []):
                    if move.get("is_best", False):
                        best_move_eval = move.get("centipawn")
                        break

            # Classify move quality based on evaluations
            curr_pos["move_quality"] = self._classify_move_quality(
                played_move_eval=played_move_eval,
                best_move_eval=best_move_eval,
                is_best_move=is_best_move,
                eval_change=player_eval_change
            )

    def _classify_move_quality(self, played_move_eval: Optional[float] = None, best_move_eval: Optional[float] = None, is_best_move: bool = False, eval_change: Optional[float] = None) -> str:
        """
        Classify move quality based on how close the played move's evaluation is to the best move's evaluation.

        Args:
            played_move_eval: Evaluation of the played move in centipawns (from player's perspective)
            best_move_eval: Evaluation of the best move in centipawns (from player's perspective)
            is_best_move: Whether the played move is one of the best moves
            eval_change: Change in evaluation after the move (player's perspective)

        Returns:
            Move quality classification
        """
        # If we know it's the best move, it's simple
        if is_best_move:
            # If eval_change is significantly negative despite being the best move,
            # it might be a forced move in a bad position
            if eval_change is not None and eval_change < -100:
                return "forced"
            return "best"

        # If we have both evaluations, compare them
        if played_move_eval is not None and best_move_eval is not None:
            # Calculate the difference between played move and best move
            # (from player's perspective, so higher is better)
            eval_diff = played_move_eval - best_move_eval

            # Classify based on the evaluation difference
            if eval_diff >= -10:  # Within 0.1 pawns of best move
                return "excellent"
            elif eval_diff >= -30:  # Within 0.3 pawns of best move
                return "good"
            elif eval_diff >= -70:  # Within 0.7 pawns of best move
                return "inaccuracy"
            elif eval_diff >= -150:  # Within 1.5 pawns of best move
                return "mistake"
            else:  # More than 1.5 pawns worse than best move
                return "blunder"

        # Fall back to eval_change if comparative evaluations are not available
        if eval_change is not None:
            if eval_change >= 0:
                return "good"  # At least didn't lose ground
            elif eval_change > -50:
                return "inaccuracy"
            elif eval_change > -150:
                return "mistake"
            else:
                return "blunder"

        # Default if no evaluation information is available
        return "normal"

    def _identify_key_positions(self, positions: List[PositionDataType]) -> List[KeyPositionType]:
        """
        Identify key positions in the game.

        Args:
            positions: List of analyzed positions

        Returns:
            List of key positions
        """
        key_positions = []

        for pos in positions:
            # Skip if missing required data
            if "eval_change" not in pos:
                continue

            # Get information needed for move quality classification
            played_move_eval = None
            best_move_eval = None
            is_best_move = False
            eval_change = pos.get("eval_change")

            if "top_moves" in pos:
                # Find the played move in top moves
                played_move = pos.get("move")
                for move in pos.get("top_moves", []):
                    if move.get("move") == played_move:
                        played_move_eval = move.get("centipawn")
                        is_best_move = move.get("is_best", False)
                        break

                # Find the best move evaluation
                for move in pos.get("top_moves", []):
                    if move.get("is_best", False):
                        best_move_eval = move.get("centipawn")
                        break

            # Classify move quality based on evaluations
            move_quality = self._classify_move_quality(
                played_move_eval=played_move_eval,
                best_move_eval=best_move_eval,
                is_best_move=is_best_move,
                eval_change=eval_change
            )

            # Include all mistakes and blunders
            if move_quality in ["mistake", "blunder"]:
                key_pos = self._create_key_position(pos)
                key_positions.append(key_pos)
                continue

            # Include excellent moves and best moves that are tactically significant
            if move_quality in ["excellent", "best"] and pos.get("position_type") == "tactical":
                key_pos = self._create_key_position(pos)
                key_positions.append(key_pos)
                continue

            # Include forced moves (often interesting)
            if move_quality == "forced":
                key_pos = self._create_key_position(pos)
                key_positions.append(key_pos)
                continue

            # Include critical opening moves
            if pos.get("position_type") == "opening" and pos["move_number"] <= 10:
                # Include if the played move wasn't the best move
                if not is_best_move:
                    key_pos = self._create_key_position(pos)
                    key_positions.append(key_pos)
                    continue

            # Include critical endgame positions
            if pos.get("position_type") == "endgame":
                # Include positions with significant material imbalance or near the end
                if "evaluation" in pos and abs(pos["evaluation"].get("value", 0)) > 200:
                    key_pos = self._create_key_position(pos)
                    key_positions.append(key_pos)

        # Limit to 10 most significant positions
        key_positions.sort(key=lambda x: abs(x["eval_change"]), reverse=True)
        return key_positions[:10]

    def _create_key_position(self, pos: Dict[str, Any]) -> KeyPositionType:
        """
        Create a key position object from a position.
        Improved implementation that correctly identifies alternative moves.

        Args:
            pos: Position data

        Returns:
            Key position object
        """
        # Generate description
        description = self._generate_position_description(pos)

        # Extract alternative moves that the player could have played
        alternative_moves = []
        if "top_moves" in pos and len(pos["top_moves"]) > 1:
            # Get the move that was actually played
            played_move = pos["move"]

            # Find better alternatives
            for move in pos["top_moves"]:
                # Only include moves that are different from what was played
                # and have better or similar evaluation
                if move["move"] != played_move:
                    # For the player's perspective, higher evaluation is better
                    alternative_moves.append({
                        "move": move["move"],
                        "evaluation": move.get("centipawn", 0),
                        "is_best": move.get("is_best", False)
                    })

            # Sort alternatives by evaluation (best first)
            alternative_moves.sort(key=lambda x: x["evaluation"], reverse=True)

        # Get information needed for move quality classification
        played_move_eval = None
        best_move_eval = None
        is_best_move = False
        eval_change = pos.get("eval_change")

        if "top_moves" in pos:
            # Find the played move in top moves
            played_move = pos.get("move")
            for move in pos.get("top_moves", []):
                if move.get("move") == played_move:
                    played_move_eval = move.get("centipawn")
                    is_best_move = move.get("is_best", False)
                    break

            # Find the best move evaluation
            for move in pos.get("top_moves", []):
                if move.get("is_best", False):
                    best_move_eval = move.get("centipawn")
                    break

        # Classify move quality based on evaluations
        move_quality = self._classify_move_quality(
            played_move_eval=played_move_eval,
            best_move_eval=best_move_eval,
            is_best_move=is_best_move,
            eval_change=eval_change
        )

        return {
            "fen": pos["fen"],
            "move_number": pos["move_number"],
            "move": pos["move"],
            "color": pos["color"],
            "eval_change": pos.get("eval_change", 0),
            "raw_eval_change": pos.get("raw_eval_change", 0),
            "move_quality": move_quality,  # Use our new classification
            "description": description,
            "alternative_moves": alternative_moves,
            "position_type": pos.get("position_type", "middlegame"),
            "played_move_rank": pos.get("played_move_rank")
        }

    def _generate_position_description(self, pos: Dict[str, Any]) -> str:
        """
        Generate a description for a position.

        Args:
            pos: Position data

        Returns:
            Position description
        """
        move_quality = pos.get("move_quality", "")
        eval_change = pos.get("eval_change", 0)
        position_type = pos.get("position_type", "")

        if move_quality == "blunder":
            return f"A serious mistake that loses approximately {abs(eval_change)/100:.1f} pawns of advantage"
        elif move_quality == "mistake":
            return f"A mistake that loses approximately {abs(eval_change)/100:.1f} pawns of advantage"
        elif move_quality == "inaccuracy":
            return f"A slight inaccuracy that loses approximately {abs(eval_change)/100:.1f} pawns of advantage"
        elif move_quality == "good":
            return f"A good move that maintains the position"
        elif move_quality == "excellent":
            return f"An excellent move that finds a strong continuation"
        elif move_quality == "best":
            if position_type == "tactical":
                return f"The best tactical move in this position"
            elif position_type == "opening":
                return f"A strong opening move that develops pieces effectively"
            elif position_type == "endgame":
                return f"A precise endgame move that maximizes winning chances"
            else:
                return f"The best move according to the engine"

        return f"Move {pos['move']} ({move_quality})"

    def _generate_analysis_summary(self, positions: List[PositionDataType]) -> AnalysisSummaryType:
        """
        Generate a summary of the analysis.

        Args:
            positions: List of analyzed positions

        Returns:
            Summary dictionary
        """
        # Filter out positions without moves (like initial position)
        valid_positions = [pos for pos in positions if pos.get("move")]

        # Count move qualities
        white_moves = [pos for pos in valid_positions if pos.get("color") == "w"]
        black_moves = [pos for pos in valid_positions if pos.get("color") == "b"]

        white_stats = self._calculate_player_stats(white_moves)
        black_stats = self._calculate_player_stats(black_moves)

        # Calculate average evaluation
        evals = [pos.get("evaluation", {}).get("value", 0)
                for pos in valid_positions if "evaluation" in pos]
        avg_eval = sum(evals) / max(1, len(evals))

        # Count moves by phase
        opening_moves = sum(1 for pos in valid_positions if pos.get("position_type") == "opening")
        middlegame_moves = sum(1 for pos in valid_positions if pos.get("position_type") == "middlegame")
        endgame_moves = sum(1 for pos in valid_positions if pos.get("position_type") == "endgame")

        # Find decisive moves (large eval changes)
        decisive_moves = [pos["move_number"] for pos in valid_positions
                         if "eval_change" in pos and abs(pos["eval_change"]) > 200]

        return {
            "white": white_stats,
            "black": black_stats,
            "total_moves": len(valid_positions),  # Count only valid moves
            "average_eval": avg_eval,
            "opening_moves": opening_moves,
            "middlegame_moves": middlegame_moves,
            "endgame_moves": endgame_moves,
            "decisive_moves": decisive_moves
        }

    def _identify_game_phases(self, positions: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Identify the phases of the game.

        Args:
            positions: List of analyzed positions

        Returns:
            Dictionary mapping phases to move numbers
        """
        phases = {
            "opening": [],
            "middlegame": [],
            "endgame": []
        }

        for pos in positions:
            phase = pos.get("position_type", "")
            move_number = pos.get("move_number", 0)

            if phase in phases:
                phases[phase].append(move_number)

        return phases

    def _calculate_player_stats(self, moves: List[PositionDataType]) -> PlayerStatsType:
        """
        Calculate statistics for a player.

        Args:
            moves: List of moves by the player

        Returns:
            Statistics dictionary
        """
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

        # Count move qualities
        blunders = sum(1 for move in moves if move.get("move_quality") == "blunder")
        mistakes = sum(1 for move in moves if move.get("move_quality") == "mistake")
        inaccuracies = sum(1 for move in moves if move.get("move_quality") == "inaccuracy")
        good_moves = sum(1 for move in moves if move.get("move_quality") == "good")
        excellent_moves = sum(1 for move in moves if move.get("move_quality") == "excellent")
        best_moves = sum(1 for move in moves if move.get("move_quality") == "best")

        # Calculate average centipawn loss
        centipawn_losses = [abs(move.get("eval_change", 0)) for move in moves
                           if "eval_change" in move and move.get("eval_change", 0) < 0]
        avg_centipawn_loss = sum(centipawn_losses) / max(1, len(centipawn_losses))

        # Calculate accuracy (improved method)
        accuracy = max(0, 100 - min(100, avg_centipawn_loss / 10))  # Better accuracy calculation
        accuracy = max(0, min(100, accuracy))  # Clamp between 0 and 100

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

    def prepare_analysis_for_llm(self, stockfish_results: StockfishAnalysisType) -> Dict[str, Any]:
        """
        Format Stockfish analysis results for LLM processing with enhanced information.
        Improved implementation with cleaner formatting and better handling of move data.
        """
        # Extract key information
        positions = stockfish_results.get("positions", [])
        key_positions = stockfish_results.get("key_positions", [])
        summary = stockfish_results.get("summary", {})
        game_phases = stockfish_results.get("game_phases", {})

        # Format game overview
        white_stats = summary.get("white", {})
        black_stats = summary.get("black", {})

        game_overview = {
            "total_moves": summary.get("total_moves", 0),
            "white_accuracy": white_stats.get("accuracy", 0),
            "black_accuracy": black_stats.get("accuracy", 0),
            "white_mistakes": white_stats.get("mistakes", 0) + white_stats.get("blunders", 0),
            "black_mistakes": black_stats.get("mistakes", 0) + black_stats.get("blunders", 0),
            "opening_moves": summary.get("opening_moves", 0),
            "middlegame_moves": summary.get("middlegame_moves", 0),
            "endgame_moves": summary.get("endgame_moves", 0),
            "decisive_moves": summary.get("decisive_moves", [])
        }

        # Format key positions with enhanced information
        formatted_key_positions = []
        for pos in key_positions:
            formatted_pos = {
                "move_number": pos["move_number"],
                "color": "White" if pos["color"] == "w" else "Black",
                "move": pos["move"],
                "quality": pos["move_quality"],
                "description": pos["description"],
                "position_type": pos["position_type"],
                "alternatives": []
            }

            # Add alternative moves
            for alt in pos.get("alternative_moves", []):
                formatted_pos["alternatives"].append({
                    "move": alt["move"],
                    "evaluation": alt["evaluation"]
                })

            formatted_key_positions.append(formatted_pos)

        # Format all moves with detailed information
        formatted_moves = []
        for pos in positions:
            # Skip positions without moves
            if not pos.get("move"):
                continue

            # Get evaluation in pawns
            eval_value = 0
            eval_type = pos.get("evaluation", {}).get("type", "")
            if eval_type == "cp":
                eval_value = pos.get("evaluation", {}).get("value", 0)
            elif eval_type == "mate":
                mate_value = pos.get("evaluation", {}).get("value", 0)
                eval_value = 999 if mate_value > 0 else -999  # Use large value for mate

            # Format move data
            move_data = {
                "move_number": pos.get("move_number", 0),
                "color": "White" if pos.get("color", "") == "w" else "Black",
                "move": pos.get("move", ""),
                "evaluation": eval_value,
                "position_type": pos.get("position_type", "middlegame"),
                "quality": pos.get("move_quality", "normal"),
                "eval_change": pos.get("eval_change", 0) if "eval_change" in pos else 0,  # Convert to pawns
                "top_engine_moves": []
            }

            # Add top engine moves
            for top_move in pos.get("top_moves", []):
                # Skip if this is the move that was played (to avoid confusion)
                if top_move.get("is_played", False):
                    continue

                move_data["top_engine_moves"].append({
                    "move": top_move.get("move", ""),
                    "evaluation": top_move.get("centipawn", 0) / 100,  # Convert to pawns
                    "is_best": top_move.get("is_best", False),
                    "move_line": top_move.get("move_line", [])
                })

            formatted_moves.append(move_data)
        print(formatted_moves)
        # Format game phases
        formatted_game_phases = {
            "opening": game_phases.get("opening", []),
            "middlegame": game_phases.get("middlegame", []),
            "endgame": game_phases.get("endgame", [])
        }

        # Return formatted analysis
        return {
            "game_overview": game_overview,
            "key_positions": formatted_key_positions,
            "game_phases": formatted_game_phases,
            "moves": formatted_moves  # Add the complete move list
        }

    def annotate_pgn(self, pgn: str, annotations: List[str]) -> str:
        """
        Annotate a PGN with comments from list of annotations.
        """
        # Parse the original PGN
        game = chess.pgn.read_game(io.StringIO(pgn))
        if not game:
            return pgn

        # Add comments to the game
        for child, annotation in zip(game.mainline(), annotations):
            child.comment = annotation
        
        # Convert back to PGN string
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        pgn_string = game.accept(exporter)

        return pgn_string


    def create_annotated_pgn(self, pgn: str, _: StockfishAnalysisType, llm_results: Dict[str, Any]) -> str:
        """
        Create an annotated PGN with comments from LLM and enhanced Stockfish analysis.
        """
        # Parse the original PGN
        game = chess.pgn.read_game(io.StringIO(pgn))
        if not game:
            return pgn

        # Create a mapping of positions to annotations
        annotations_map = {}
        for annotation in llm_results.get("annotations", []):
            move_number = annotation.get("move_number", 0)
            color = annotation.get("color", "w")
            key = f"{move_number}{'.' if color == 'w' else '...'}"

            # Create enhanced annotation with evaluation
            comment = annotation.get("comment", "")
            evaluation = annotation.get("evaluation", "")
            category = annotation.get("category", "")

            # Add evaluation symbol if provided
            if evaluation:
                comment = f"{evaluation} {comment}"

            # Add category if provided
            if category:
                comment = f"[{category}] {comment}"

            annotations_map[key] = comment

        # Add comments to the game
        move_number = 1
        ply = 0

        for child in game.mainline():
            ply += 1
            color = "w" if ply % 2 == 1 else "b"
            key = f"{move_number}{'.' if color == 'w' else '...'}"

            # Add annotation if available
            if key in annotations_map:
                child.comment = annotations_map[key]
            else:
                # If no annotation is provided for this move, add a default one
                # This should not happen if the LLM is properly annotating every move
                move = child.san()
                child.comment = f"[auto] {move}"
                print(f"Warning: Missing annotation for move {key} {move}")

            if ply % 2 == 0:  # After Black's move
                move_number += 1

        # Convert back to PGN string
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        pgn_string = game.accept(exporter)

        # Verify all moves have annotations
        moves_count = len(list(game.mainline()))
        annotations_count = len(llm_results.get("annotations", []))
        if moves_count != annotations_count:
            print(f"Warning: Number of moves ({moves_count}) does not match number of annotations ({annotations_count})")

        return pgn_string
