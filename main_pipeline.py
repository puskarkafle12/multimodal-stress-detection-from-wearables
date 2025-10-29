"""
Main Pipeline Script
===================

This is the main script that orchestrates the entire stress prediction pipeline.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import argparse
import json
from typing import Dict, List, Optional

# Import all pipeline modules
from stress_pipeline import PipelineConfig
from enhanced_preprocessing_data_ingestion import EnhancedPreprocessingDataIngestion
from windowing import WindowingSystem
from data_splits import DataSplitter, StressDataset
from models import ModelFactory
from training import StressTrainer, GroupAwareTrainer
from inference import StressInference, ParticipantLevelAnalyzer
from evaluation import StressEvaluator
from interpretability import ModelInterpretability, RobustnessTester, AblationStudy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stress_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StressPredictionPipeline:
    """Main pipeline class that orchestrates the entire process"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.data_ingestion = EnhancedPreprocessingDataIngestion(config)
        self.windowing_system = WindowingSystem(config)
        self.data_splitter = DataSplitter(config)
        self.evaluator = StressEvaluator(config.output_dir)
        self.analyzer = ParticipantLevelAnalyzer(config.output_dir)
        
        logger.info(f"Pipeline initialized with device: {self.device}")
        logger.info(f"Output directory: {config.output_dir}")
    
    def run_full_pipeline(self, participant_limit: Optional[int] = None):
        """Run the complete pipeline"""
        logger.info("Starting full stress prediction pipeline...")
        
        # Step 1: Data Ingestion and Harmonization
        logger.info("Step 1: Data Ingestion and Harmonization")
        all_windows = self._ingest_and_harmonize_data(participant_limit)
        
        if not all_windows:
            logger.error("No data could be processed. Exiting.")
            return
        
        # Step 2: Create Data Splits
        logger.info("Step 2: Creating Data Splits")
        train_windows, val_windows, test_windows = self._create_data_splits(all_windows)
        
        # Step 3: Create DataLoaders
        logger.info("Step 3: Creating DataLoaders")
        train_loader, val_loader, test_loader = self.data_splitter.create_dataloaders(
            train_windows, val_windows, test_windows
        )
        
        # Step 4: Model Training
        logger.info("Step 4: Model Training")
        model = self._train_model(train_loader, val_loader)
        
        # Step 5: Inference and Aggregation
        logger.info("Step 5: Inference and Participant-Level Aggregation")
        participant_metrics = self._run_inference(model, test_loader)
        
        # Step 6: Evaluation
        logger.info("Step 6: Comprehensive Evaluation")
        self._run_evaluation(model, test_loader, participant_metrics)
        
        # Step 7: Interpretability Analysis
        logger.info("Step 7: Interpretability Analysis")
        self._run_interpretability_analysis(model, test_loader)
        
        logger.info("Pipeline completed successfully!")
    
    def _ingest_and_harmonize_data(self, participant_limit: Optional[int] = None) -> List[Dict]:
        """Ingest and harmonize data from all participants"""
        all_windows = []
        
        # Get available participants
        participants = self.data_ingestion.get_available_participants()
        
        if participant_limit:
            participants = participants[:participant_limit]
        
        logger.info(f"Processing {len(participants)} participants...")
        
        for i, participant_id in enumerate(participants):
            logger.info(f"Processing participant {participant_id} ({i+1}/{len(participants)})")
            
            try:
                # Load participant streams
                streams = self.data_ingestion.load_participant_streams(participant_id)
                
                if not streams:
                    logger.warning(f"No streams found for participant {participant_id}")
                    continue
                
                synced_streams = self.data_ingestion.resample_and_sync_streams(streams, self.config.sampling_rate)
                
                if not synced_streams:
                    logger.warning(f"No synced streams for participant {participant_id}")
                    continue
                
                # Create windows
                windows = self.windowing_system.create_windows(synced_streams, participant_id)
                
                if windows:
                    all_windows.extend(windows)
                    logger.info(f"Created {len(windows)} windows for participant {participant_id}")
                else:
                    logger.warning(f"No windows created for participant {participant_id}")
                    
            except Exception as e:
                logger.error(f"Error processing participant {participant_id}: {e}")
                continue
        
        logger.info(f"Total windows created: {len(all_windows)}")
        return all_windows
    
    def _create_data_splits(self, all_windows: List[Dict]) -> tuple:
        """Create train/val/test splits"""
        participants_df = self.data_ingestion.participants_df
        
        train_windows, val_windows, test_windows = self.data_splitter.create_splits(
            all_windows, participants_df
        )
        
        # Log split statistics
        split_stats = self.data_splitter.get_split_statistics(train_windows, val_windows, test_windows)
        logger.info(f"Split statistics: {split_stats}")
        
        return train_windows, val_windows, test_windows
    
    def _train_model(self, train_loader, val_loader) -> torch.nn.Module:
        """Train the stress prediction model"""
        # Determine input dimension
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch['features'].shape[2]  # [batch, time, features]
        
        # Create model
        model = ModelFactory.create_model(self.config.model_type, input_dim, self.config)
        
        # Create trainer
        if self.config.model_type == "multimodal":
            # Use group-aware trainer for multimodal models
            trainer = GroupAwareTrainer(
                model, self.config, self.config.output_dir, 
                self.data_ingestion.participants_df
            )
        else:
            trainer = StressTrainer(model, self.config, self.config.output_dir)
        
        # Train model
        training_history = trainer.train(train_loader, val_loader)
        
        return model
    
    def _run_inference(self, model: torch.nn.Module, test_loader) -> Dict[str, Dict[str, float]]:
        """Run inference and participant-level aggregation"""
        # Check if test loader has data
        if len(test_loader) == 0:
            logger.warning("No test data available for inference. Skipping inference step.")
            return {}
        
        # Create inference object
        inference = StressInference(model, self.device)
        
        # Predict windows
        predictions_by_participant = inference.predict_windows(test_loader)
        
        # Aggregate to participant level
        participant_metrics = inference.aggregate_participant_predictions(
            predictions_by_participant, self.config.stress_threshold
        )
        
        # Create summary
        participants_df = self.data_ingestion.participants_df
        summary_df = self.analyzer.create_participant_summary(participant_metrics, participants_df)
        
        # Plot results
        self.analyzer.plot_participant_distributions(summary_df, 
            str(self.config.output_dir / "participant_distributions.png"))
        
        # Save results
        self.analyzer.save_results(participant_metrics, summary_df, {}, {})
        
        return participant_metrics
    
    def _run_evaluation(self, model: torch.nn.Module, test_loader, 
                        participant_metrics: Dict[str, Dict[str, float]]):
        """Run comprehensive evaluation"""
        # Check if test loader has data
        if len(test_loader) == 0:
            logger.warning("No test data available for evaluation. Skipping evaluation step.")
            return
        
        # Get true labels and predictions
        y_true, y_pred, participant_ids = self._get_predictions_and_labels(model, test_loader)
        
        # Window-level evaluation
        window_metrics = self.evaluator.evaluate_window_level(y_true, y_pred)
        
        # Participant-level evaluation
        participants_df = self.data_ingestion.participants_df
        participant_eval_metrics = self.evaluator.evaluate_participant_level(
            participant_metrics, participants_df
        )
        
        # Temporal evaluation
        predictions_by_participant = self._group_predictions_by_participant(y_pred, participant_ids)
        temporal_metrics = self.evaluator.evaluate_temporal_consistency(predictions_by_participant)
        
        # Subgroup evaluation
        subgroup_metrics = self.evaluator.evaluate_subgroup_performance(
            y_true, y_pred, participant_ids, participants_df
        )
        
        # Plot results
        self.evaluator.plot_evaluation_results(
            window_metrics, participant_eval_metrics, subgroup_metrics,
            str(self.config.output_dir / "evaluation_results.png")
        )
        
        # Save results
        self.evaluator.save_evaluation_results(
            window_metrics, participant_eval_metrics, temporal_metrics, subgroup_metrics
        )
    
    def _run_interpretability_analysis(self, model: torch.nn.Module, test_loader):
        """Run interpretability and robustness analysis"""
        # Check if test loader has data
        if len(test_loader) == 0:
            logger.warning("No test data available for interpretability analysis. Skipping interpretability step.")
            return
        
        # Create interpretability object
        interpretability = ModelInterpretability(model, self.device, self.config.output_dir)
        
        # Compute feature importance
        feature_importance = interpretability.compute_feature_importance(test_loader)
        modality_importance = interpretability.compute_modality_importance(test_loader)
        
        # Plot importance
        interpretability.plot_feature_importance(feature_importance,
            str(self.config.output_dir / "feature_importance.png"))
        interpretability.plot_modality_importance(modality_importance,
            str(self.config.output_dir / "modality_importance.png"))
        
        # Robustness testing
        robustness_tester = RobustnessTester(model, self.device, self.config.output_dir)
        
        missingness_results = robustness_tester.test_missingness_robustness(test_loader)
        resolution_results = robustness_tester.test_resolution_robustness(test_loader)
        alignment_results = robustness_tester.test_alignment_robustness(test_loader)
        window_size_results = robustness_tester.test_window_size_robustness(test_loader)
        
        # Plot robustness results
        robustness_tester.plot_robustness_results(
            missingness_results, resolution_results, alignment_results, window_size_results,
            str(self.config.output_dir / "robustness_results.png")
        )
        
        # Save robustness results
        robustness_tester.save_robustness_results(
            missingness_results, resolution_results, alignment_results, window_size_results
        )
        
        # Ablation study
        ablation_study = AblationStudy(model, self.device, self.config.output_dir)
        
        modality_ablation = ablation_study.conduct_modality_ablation(test_loader)
        feature_ablation = ablation_study.conduct_feature_ablation(test_loader)
        
        # Plot ablation results
        ablation_study.plot_ablation_results(modality_ablation, feature_ablation,
            str(self.config.output_dir / "ablation_results.png"))
        
        # Save ablation results
        ablation_study.save_ablation_results(modality_ablation, feature_ablation)
    
    def _get_predictions_and_labels(self, model: torch.nn.Module, test_loader) -> tuple:
        """Get predictions and labels from test data"""
        model.eval()
        y_true = []
        y_pred = []
        participant_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                pids = batch['participant_id']
                
                predictions = model(features)
                
                y_true.extend(labels.cpu().numpy().flatten())
                y_pred.extend(predictions.cpu().numpy().flatten())
                participant_ids.extend(pids)
        
        return np.array(y_true), np.array(y_pred), participant_ids
    
    def _group_predictions_by_participant(self, y_pred: np.ndarray, participant_ids: List[str]) -> Dict[str, List[float]]:
        """Group predictions by participant"""
        predictions_by_participant = {}
        
        for pred, pid in zip(y_pred, participant_ids):
            if pid not in predictions_by_participant:
                predictions_by_participant[pid] = []
            predictions_by_participant[pid].append(pred)
        
        return predictions_by_participant

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='AI-READI Stress Prediction Pipeline')
    parser.add_argument('--data_root', type=str, 
                       default='/Users/puskarkafle/Documents/Research/AI-READI',
                       help='Root directory of AI-READI data')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/puskarkafle/Documents/Research/stress_pipeline_output',
                       help='Output directory for results')
    parser.add_argument('--model_type', type=str, default='cnn',
                       choices=['cnn', 'tcn', 'transformer', 'multimodal'],
                       help='Model architecture to use')
    parser.add_argument('--window_length', type=int, default=10,
                       help='Window length in minutes')
    parser.add_argument('--stride', type=int, default=2,
                       help='Stride in minutes')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--participant_limit', type=int, default=None,
                       help='Limit number of participants to process')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        window_length_min=args.window_length,
        stride_min=args.stride,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_type=args.model_type
    )
    
    # Create and run pipeline
    pipeline = StressPredictionPipeline(config)
    pipeline.run_full_pipeline(args.participant_limit)

if __name__ == "__main__":
    main()
