package sim.aggregation.random.collision.avoidance;

import java.awt.Color;

import javax.swing.Box;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sim.graphics.Window;
import sim.graphics.World;

@SuppressWarnings("serial")
public class A1Window extends Window{
	int minA=1,maxA=100,numA=20;
	int minB=1,maxB=10,numB=10;
	int minSight=5,maxSight=100,numSight=25;
	int minBias=1,maxBias=1000,numBias=100;
	int x=1000,y=700;
	
	JPanel panel;
	JSlider sliderA,sliderB,sliderS,sliderBias;
	JLabel labelA,labelB,labelS,labelBias;
	
	public World createWorld() {
		World world = new World(x,y);
		world.enableDrawing(Color.GREEN, numB);
		for(int i=0;i<numA;i++){
			world.addObject(new ACABot(numB,numSight,numBias));
		}
		return world;
	}

	@Override
	public JPanel getSettings() {
		panel = new JPanel();
		Box settingsBox = Box.createVerticalBox();		
		settingsBox.setAlignmentX(CENTER_ALIGNMENT);
		settingsBox.setAlignmentY(CENTER_ALIGNMENT);
		
		sliderA = new JSlider();
		labelA = new JLabel("Agent Number: "+numA);
		////////////////////////////////////////////////////
		settingsBox.add(labelA);
		sliderA.setMinimum(minA);
		sliderA.setMaximum(maxA);
		sliderA.setValue(numA);
		sliderA.addChangeListener(new ChangeListener()
		{
			public void stateChanged(final ChangeEvent event)
			{
				agentsChange(event);
			}
		});
		
		settingsBox.add(sliderA);
		/////////////////////////////////////////////////////
		sliderB = new JSlider();
		labelB = new JLabel("Agent Size: "+numB);
		settingsBox.add(labelB);
		sliderB.setMinimum(minB);
		sliderB.setMaximum(maxB);
		sliderB.setValue(numB);
		sliderB.addChangeListener(new ChangeListener()
		{
			public void stateChanged(ChangeEvent e) {
				agentsSizeChange(e);
			}
			
		});
		settingsBox.add(sliderB);
		
		////////////////////////////////////////////////////
		sliderS = new JSlider();
		labelS = new JLabel("Agent's Sight Range: "+numSight);
		//settingsBox.add(labelS);
		sliderS.setMinimum(minSight);
		sliderS.setMaximum(maxSight);
		sliderS.setValue(numSight);
		sliderS.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				agentsSightChange(e);
			}
			
		});
		//settingsBox.add(sliderS);
		
		sliderBias = new JSlider();
		labelBias = new JLabel("Agent's Bias: "+numBias);
		settingsBox.add(labelBias);
		sliderBias.setMinimum(minBias);
		sliderBias.setMaximum(maxBias);
		sliderBias.setValue(numBias);
		sliderBias.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				agentsBiasChange(e);
			}
			
		});
		settingsBox.add(sliderBias);
		
		panel.add(settingsBox);
		return panel;
	}

	public void agentsChange(ChangeEvent event){
		numA=sliderA.getValue();
		labelA.setText("Agent Number: "+numA);
	}
	
	public void agentsSizeChange(ChangeEvent event){
		numB=sliderB.getValue();
		labelB.setText("Agent Size: "+numB);
	}
	
	public void agentsSightChange(ChangeEvent event){
		numSight=sliderS.getValue();
		labelS.setText("Agent's Sight Range: "+numSight);
	}
	
	public void agentsBiasChange(ChangeEvent event){
		numBias=sliderBias.getValue();
		labelBias.setText("Agent's Bias: "+numBias);
	}
}
