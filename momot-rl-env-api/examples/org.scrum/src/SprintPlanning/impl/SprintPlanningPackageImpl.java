/**
 */
package SprintPlanning.impl;

import SprintPlanning.Backlog;
import SprintPlanning.Plan;
import SprintPlanning.Sprint;
import SprintPlanning.SprintPlanningFactory;
import SprintPlanning.SprintPlanningPackage;
import SprintPlanning.Stakeholder;
import SprintPlanning.WorkItem;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

import org.eclipse.emf.ecore.impl.EPackageImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Package</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class SprintPlanningPackageImpl extends EPackageImpl implements SprintPlanningPackage {
   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   private EClass stakeholderEClass = null;

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   private EClass backlogEClass = null;

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   private EClass workItemEClass = null;

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   private EClass sprintEClass = null;

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   private EClass planEClass = null;

   /**
    * Creates an instance of the model <b>Package</b>, registered with
    * {@link org.eclipse.emf.ecore.EPackage.Registry EPackage.Registry} by the package
    * package URI value.
    * <p>Note: the correct way to create the package is via the static
    * factory method {@link #init init()}, which also performs
    * initialization of the package, or returns the registered package,
    * if one already exists.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @see org.eclipse.emf.ecore.EPackage.Registry
    * @see SprintPlanning.SprintPlanningPackage#eNS_URI
    * @see #init()
    * @generated
    */
   private SprintPlanningPackageImpl() {
      super(eNS_URI, SprintPlanningFactory.eINSTANCE);
   }
   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   private static boolean isInited = false;

   /**
    * Creates, registers, and initializes the <b>Package</b> for this model, and for any others upon which it depends.
    *
    * <p>This method is used to initialize {@link SprintPlanningPackage#eINSTANCE} when that field is accessed.
    * Clients should not invoke it directly. Instead, they should simply access that field to obtain the package.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @see #eNS_URI
    * @see #createPackageContents()
    * @see #initializePackageContents()
    * @generated
    */
   public static SprintPlanningPackage init() {
      if (isInited) return (SprintPlanningPackage)EPackage.Registry.INSTANCE.getEPackage(SprintPlanningPackage.eNS_URI);

      // Obtain or create and register package
      Object registeredSprintPlanningPackage = EPackage.Registry.INSTANCE.get(eNS_URI);
      SprintPlanningPackageImpl theSprintPlanningPackage = registeredSprintPlanningPackage instanceof SprintPlanningPackageImpl ? (SprintPlanningPackageImpl)registeredSprintPlanningPackage : new SprintPlanningPackageImpl();

      isInited = true;

      // Create package meta-data objects
      theSprintPlanningPackage.createPackageContents();

      // Initialize created meta-data
      theSprintPlanningPackage.initializePackageContents();

      // Mark meta-data to indicate it can't be changed
      theSprintPlanningPackage.freeze();

      // Update the registry and return the package
      EPackage.Registry.INSTANCE.put(SprintPlanningPackage.eNS_URI, theSprintPlanningPackage);
      return theSprintPlanningPackage;
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EClass getStakeholder() {
      return stakeholderEClass;
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EReference getStakeholder_Workitem() {
      return (EReference)stakeholderEClass.getEStructuralFeatures().get(0);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EClass getBacklog() {
      return backlogEClass;
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EReference getBacklog_Workitems() {
      return (EReference)backlogEClass.getEStructuralFeatures().get(0);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EClass getWorkItem() {
      return workItemEClass;
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EReference getWorkItem_IsPlannedFor() {
      return (EReference)workItemEClass.getEStructuralFeatures().get(0);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EAttribute getWorkItem_Importance() {
      return (EAttribute)workItemEClass.getEStructuralFeatures().get(1);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EAttribute getWorkItem_Effort() {
      return (EAttribute)workItemEClass.getEStructuralFeatures().get(2);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EReference getWorkItem_Stakeholder() {
      return (EReference)workItemEClass.getEStructuralFeatures().get(3);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EAttribute getWorkItem_Name() {
      return (EAttribute)workItemEClass.getEStructuralFeatures().get(4);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EClass getSprint() {
      return sprintEClass;
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EReference getSprint_CommittedItem() {
      return (EReference)sprintEClass.getEStructuralFeatures().get(0);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EAttribute getSprint_Name() {
      return (EAttribute)sprintEClass.getEStructuralFeatures().get(1);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EClass getPlan() {
      return planEClass;
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EReference getPlan_Stakeholders() {
      return (EReference)planEClass.getEStructuralFeatures().get(0);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EReference getPlan_Backlog() {
      return (EReference)planEClass.getEStructuralFeatures().get(1);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EReference getPlan_Sprints() {
      return (EReference)planEClass.getEStructuralFeatures().get(2);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EAttribute getPlan_MaxSprints() {
      return (EAttribute)planEClass.getEStructuralFeatures().get(3);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EAttribute getPlan_MinSprints() {
      return (EAttribute)planEClass.getEStructuralFeatures().get(4);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public EAttribute getPlan_CurrentSprints() {
      return (EAttribute)planEClass.getEStructuralFeatures().get(5);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public SprintPlanningFactory getSprintPlanningFactory() {
      return (SprintPlanningFactory)getEFactoryInstance();
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   private boolean isCreated = false;

   /**
    * Creates the meta-model objects for the package.  This method is
    * guarded to have no affect on any invocation but its first.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public void createPackageContents() {
      if (isCreated) return;
      isCreated = true;

      // Create classes and their features
      stakeholderEClass = createEClass(STAKEHOLDER);
      createEReference(stakeholderEClass, STAKEHOLDER__WORKITEM);

      backlogEClass = createEClass(BACKLOG);
      createEReference(backlogEClass, BACKLOG__WORKITEMS);

      workItemEClass = createEClass(WORK_ITEM);
      createEReference(workItemEClass, WORK_ITEM__IS_PLANNED_FOR);
      createEAttribute(workItemEClass, WORK_ITEM__IMPORTANCE);
      createEAttribute(workItemEClass, WORK_ITEM__EFFORT);
      createEReference(workItemEClass, WORK_ITEM__STAKEHOLDER);
      createEAttribute(workItemEClass, WORK_ITEM__NAME);

      sprintEClass = createEClass(SPRINT);
      createEReference(sprintEClass, SPRINT__COMMITTED_ITEM);
      createEAttribute(sprintEClass, SPRINT__NAME);

      planEClass = createEClass(PLAN);
      createEReference(planEClass, PLAN__STAKEHOLDERS);
      createEReference(planEClass, PLAN__BACKLOG);
      createEReference(planEClass, PLAN__SPRINTS);
      createEAttribute(planEClass, PLAN__MAX_SPRINTS);
      createEAttribute(planEClass, PLAN__MIN_SPRINTS);
      createEAttribute(planEClass, PLAN__CURRENT_SPRINTS);
   }

   /**
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   private boolean isInitialized = false;

   /**
    * Complete the initialization of the package and its meta-model.  This
    * method is guarded to have no affect on any invocation but its first.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   public void initializePackageContents() {
      if (isInitialized) return;
      isInitialized = true;

      // Initialize package
      setName(eNAME);
      setNsPrefix(eNS_PREFIX);
      setNsURI(eNS_URI);

      // Create type parameters

      // Set bounds for type parameters

      // Add supertypes to classes

      // Initialize classes, features, and operations; add parameters
      initEClass(stakeholderEClass, Stakeholder.class, "Stakeholder", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
      initEReference(getStakeholder_Workitem(), this.getWorkItem(), this.getWorkItem_Stakeholder(), "workitem", null, 1, -1, Stakeholder.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

      initEClass(backlogEClass, Backlog.class, "Backlog", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
      initEReference(getBacklog_Workitems(), this.getWorkItem(), null, "workitems", null, 1, -1, Backlog.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

      initEClass(workItemEClass, WorkItem.class, "WorkItem", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
      initEReference(getWorkItem_IsPlannedFor(), this.getSprint(), this.getSprint_CommittedItem(), "isPlannedFor", null, 0, -1, WorkItem.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
      initEAttribute(getWorkItem_Importance(), ecorePackage.getEInt(), "Importance", null, 1, 1, WorkItem.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
      initEAttribute(getWorkItem_Effort(), ecorePackage.getEInt(), "Effort", null, 1, 1, WorkItem.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
      initEReference(getWorkItem_Stakeholder(), this.getStakeholder(), this.getStakeholder_Workitem(), "stakeholder", null, 1, 1, WorkItem.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
      initEAttribute(getWorkItem_Name(), ecorePackage.getEString(), "name", null, 0, 1, WorkItem.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

      initEClass(sprintEClass, Sprint.class, "Sprint", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
      initEReference(getSprint_CommittedItem(), this.getWorkItem(), this.getWorkItem_IsPlannedFor(), "committedItem", null, 1, -1, Sprint.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
      initEAttribute(getSprint_Name(), ecorePackage.getEString(), "name", null, 0, 1, Sprint.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

      initEClass(planEClass, Plan.class, "Plan", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
      initEReference(getPlan_Stakeholders(), this.getStakeholder(), null, "stakeholders", null, 0, -1, Plan.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
      initEReference(getPlan_Backlog(), this.getBacklog(), null, "backlog", null, 1, 1, Plan.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
      initEReference(getPlan_Sprints(), this.getSprint(), null, "sprints", null, 0, -1, Plan.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
      initEAttribute(getPlan_MaxSprints(), ecorePackage.getEInt(), "maxSprints", null, 1, 1, Plan.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
      initEAttribute(getPlan_MinSprints(), ecorePackage.getEInt(), "minSprints", null, 1, 1, Plan.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
      initEAttribute(getPlan_CurrentSprints(), ecorePackage.getEInt(), "currentSprints", null, 1, 1, Plan.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

      // Create resource
      createResource(eNS_URI);

      // Create annotations
      // http://www.eclipse.org/OCL/Import
      createImportAnnotations();
   }

   /**
    * Initializes the annotations for <b>http://www.eclipse.org/OCL/Import</b>.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @generated
    */
   protected void createImportAnnotations() {
      String source = "http://www.eclipse.org/OCL/Import";
      addAnnotation
        (this,
         source,
         new String[] {
            "ecore", "http://www.eclipse.org/emf/2002/Ecore"
         });
   }

} //SprintPlanningPackageImpl
