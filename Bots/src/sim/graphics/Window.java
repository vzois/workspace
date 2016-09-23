package sim.graphics;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Hashtable;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.swing.Box;
import javax.swing.JApplet;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;


public abstract class Window extends JApplet{

	private static final long serialVersionUID = -2353424288860371544L;
	private JButton start;
	private JButton stop;
	private JButton reset;
	private JPanel buttons;
	private World world;
	
	private static Logger logger = Logger.getLogger(Window.class.getName());
	public JTextField tf;
	public JSlider speedSlider;
	public JLabel speedLabel;
	public int maxTime = 300, minTime = 10, numTime = 100;
	
	public boolean MULTI_THREADED=true;
	public JCheckBox cbM;
	public JCheckBox cbV;
	public JCheckBox fw;
	public static boolean flatWorld = false;
	public static boolean draw_velocity=false;
	
	
	public Window(){
		
	}
	
	public void init(){
		getContentPane().setLayout(new BorderLayout(2,2));
				
		start = new JButton();
		start.setText("Start");
		start.addActionListener( new ActionListener(){
			public void actionPerformed(ActionEvent ae) {
				logger.log(Level.INFO,"Start Clicked!!!");
				startButton(ae);
			}
			
		});
		start.setEnabled(true);
		
		stop = new JButton();
		stop.setText("Stop");
		stop.addActionListener( new ActionListener(){
			public void actionPerformed(ActionEvent ae) {
				logger.log(Level.INFO,"Stop Clicked!!!");
				stopButton(ae);
			}
			
		});
		stop.setEnabled(false);
		
		reset = new JButton();
		reset.setText("Reset");
		reset.addActionListener( new ActionListener(){
			public void actionPerformed(ActionEvent ae) {
				logger.log(Level.INFO,"Reset Clicked!!!");
				resetButton(ae);
			}
			
		});
		
		buttons = new JPanel();
		buttons.setLayout(new GridLayout(1,3));
		//buttons.setAlignmentY(0.5f);
		start.setPreferredSize(new Dimension(20,40));
		stop.setPreferredSize(new Dimension(20,40));
		reset.setPreferredSize(new Dimension(20,40));
		buttons.add(start);
		buttons.add(stop);
		buttons.add(reset);
		
		Box buttonsBox = Box.createHorizontalBox();
		buttonsBox.add(start);
		buttonsBox.add(stop);
		buttonsBox.add(reset);
		getContentPane().add(buttonsBox,BorderLayout.SOUTH);
		
		world = createWorld();
		tf = new JTextField();
		tf.setText(world.time+"");
		tf.setEditable(false);
		tf.setMaximumSize(new Dimension(100,25));
		tf.setMinimumSize(new Dimension(100,25));
		world.setTextField(tf);
		
		speedSlider = new JSlider();
		speedLabel = new JLabel("Agent's Bias: "+0);
		speedSlider.setMaximumSize(new Dimension(world.getLX(),world.getLY()));
		Hashtable<Integer,JLabel> labelTable = new Hashtable<Integer,JLabel>();
		speedSlider.setMinimum(minTime);
		speedSlider.setMaximum(maxTime);
		speedSlider.setValue(numTime);
		world.setSpeed(numTime);
		labelTable.put( new Integer( speedSlider.getMinimum() ), new JLabel("Fast") );
		labelTable.put( new Integer( speedSlider.getMaximum() ), new JLabel("Slow") );
		speedSlider.setLabelTable( labelTable );
		speedSlider.setPaintLabels(true);
		speedSlider.addChangeListener(new ChangeListener()
		{
			public void stateChanged(ChangeEvent e) {
				numTime = speedSlider.getValue();
				world.setSpeed(numTime);
			}
		});
		speedSlider.setMaximumSize(new Dimension(400,50));
		speedSlider.setMinimumSize(new Dimension(400,50));
		speedSlider.setSize(new Dimension(400,50));
		
		cbM = new JCheckBox("Multi - Thread Execution",MULTI_THREADED);
		cbV = new JCheckBox("Draw Agent Velocity",Window.draw_velocity);
		cbV.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) { draw_velocity(e); }
		});
		
		fw = new JCheckBox("Flat World",Window.flatWorld);
		fw.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) { flatWorld(e); }
		});
		
		Box worldBox = Box.createVerticalBox();
		worldBox.add(world);
		worldBox.setPreferredSize(new Dimension(world.getLX(),world.getLY()));
		worldBox.setAlignmentX(CENTER_ALIGNMENT);
		worldBox.setAlignmentY(CENTER_ALIGNMENT);
		worldBox.add(tf);
		worldBox.add(speedSlider);
		//worldBox.add(worldSizeSlider);
		
		Box checkBox = Box.createHorizontalBox();
		checkBox.setAlignmentX(CENTER_ALIGNMENT);
		checkBox.setAlignmentY(CENTER_ALIGNMENT);
		checkBox.add(cbM);
		checkBox.add(cbV);
		checkBox.add(fw);
		worldBox.add(checkBox);
		
		Box sideBox = Box.createVerticalBox();
		sideBox.setAlignmentX(CENTER_ALIGNMENT);
		sideBox.setAlignmentY(CENTER_ALIGNMENT);
		Window.draw_velocity=false;
		sideBox.add(getSettings());
		
		getContentPane().add(worldBox,BorderLayout.CENTER);
		getContentPane().add(sideBox,BorderLayout.EAST);
	}
	
	public void draw_velocity(ChangeEvent e){
		Window.draw_velocity = cbV.isSelected();
	}
	
	public void flatWorld(ChangeEvent e){
		Window.flatWorld = fw.isSelected();
	}
	
	private void startButton(ActionEvent ae){
		start.setEnabled(false);
		stop.setEnabled(true);
		cbM.setEnabled(false);
		if(cbM.isSelected()){
			logger.log(Level.INFO, "RUNNING BALANCED THREAD");
			world.simTime();
		}else{
			logger.log(Level.INFO, "RUNNING MULTI THREAD");
			world.simTime3();
		}
	}
	
	private void stopButton(ActionEvent ae){
		start.setEnabled(true);
		stop.setEnabled(false);
		cbM.setEnabled(true);
		if(cbM.isSelected()){
			world.freezeTime();
		}else{
			world.freezeTime3();
		}
	}
	
	private void resetButton(ActionEvent ae){
		if(start.isEnabled()){
			world.setWorld(createWorld());
			world.repaint();
		}
	}
	public JTextField getTextField(){
		return tf;
	}
	
	public abstract World createWorld();
	
	public abstract JPanel getSettings();
		
}
