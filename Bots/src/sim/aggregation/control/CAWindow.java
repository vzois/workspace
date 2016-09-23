package sim.aggregation.control;

import javax.swing.Box;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sim.graphics.Window;
import sim.graphics.World;

@SuppressWarnings("serial")
public class CAWindow extends Window {
	int minA=1,maxA=10000,numA=10;
	int minB=3,maxB=10,numB=5;
	int minSight=5,maxSight=1000,numSight=25;
	int x=700,y=700;
	
	double aF=25,bF=75,cF=75;
	double minF=1,maxF=100;
	
	
	JPanel panel;
	JSlider sliderA,sliderB,sliderS;
	JSlider sliderAF,sliderBF,sliderCF;
	JLabel labelA,labelB,labelS;
	JLabel labelAF,labelBF,labelCF;
	
	public World createWorld() {
		World world = new World(x,y);
		for(int i=0;i<numA;i++){
			world.addObject(new CABot(numB,numSight,aF,bF,cF));
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
		settingsBox.add(labelS);
		sliderS.setMinimum(minSight);
		sliderS.setMaximum(maxSight);
		sliderS.setValue(numSight);
		sliderS.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				agentsSightChange(e);
			}
			
		});
		settingsBox.add(sliderS);
		
		/////////////////////////////////////////////////////
		sliderAF = new JSlider();
		labelAF = new JLabel("Factor A: "+aF);
		settingsBox.add(labelAF);
		sliderAF.setMinimum((int)minF);
		sliderAF.setMaximum((int)maxF);
		sliderAF.setValue((int)aF);
		sliderAF.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				aFactorChange(e);
			}
			
		});
		settingsBox.add(sliderAF);
		
		/////////////////////////////////////////////////////
		sliderBF = new JSlider();
		labelBF = new JLabel("Factor B: "+bF);
		settingsBox.add(labelBF);
		sliderBF.setMinimum((int)minF);
		sliderBF.setMaximum((int)maxF);
		sliderBF.setValue((int)bF);
		sliderBF.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				bFactorChange(e);
			}
			
		});
		settingsBox.add(sliderBF);
		
		/////////////////////////////////////////////////////
		sliderCF = new JSlider();
		labelCF = new JLabel("Factor C: "+cF);
		settingsBox.add(labelCF);
		sliderCF.setMinimum((int)minF);
		sliderCF.setMaximum((int)maxF);
		sliderCF.setValue((int)cF);
		sliderCF.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				cFactorChange(e);
			}
			
		});
		settingsBox.add(sliderCF);
		
		panel.add(settingsBox);
		return panel;
	}

	public void aFactorChange(ChangeEvent event){
		aF = sliderAF.getValue();
		labelAF.setText("Factor A: "+aF);
	}
	
	public void bFactorChange(ChangeEvent event){
		bF = sliderBF.getValue();
		labelBF.setText("Factor B: "+bF);
	}

	public void cFactorChange(ChangeEvent event){
		cF = sliderCF.getValue();
		labelCF.setText("Factor C: "+cF);
	}
	
	public void agentsChange(ChangeEvent event){
		numA=sliderA.getValue();
		labelA.setText("Agent Number: "+sliderA.getValue());
	}
	
	public void agentsSizeChange(ChangeEvent event){
		numB=sliderB.getValue();
		labelB.setText("Agent Size: "+numB);
	}
	
	public void agentsSightChange(ChangeEvent event){
		numSight=sliderS.getValue();
		labelS.setText("Agent's Sight Range: "+numSight);
	}
}
